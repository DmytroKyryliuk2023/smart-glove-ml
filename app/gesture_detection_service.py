"""
Service для обробки та розпізнавання жестів з потоку даних.
Займається групуванням точок, детектуванням меж жестів та координацією з моделями.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .division_service import DivisionService
from .gesture_service import GestureService


class GestureDetectionService:
    """Сервіс для детектування та розпізнавання жестів"""

    def __init__(
        self,
        division_service: DivisionService,
        gesture_service: GestureService,
        close_points_threshold: int = 30,
        min_gesture_length: int = 100,
    ):
        self.division_service = division_service
        self.gesture_service = gesture_service
        self.close_points_threshold = close_points_threshold
        self.min_gesture_length = min_gesture_length

    def group_points(self, points: List[int], threshold: int) -> List[List[int]]:
        """
        Групує сусідні точки з різних вікон, що вказують на одну подію.

        Args:
            points: Список індексів точок
            threshold: Максимальна відстань між точками в одній групі

        Returns:
            Список груп точок
        """
        if not points:
            return []

        groups = [[points[0]]]

        for p in points[1:]:
            if p - groups[-1][-1] <= threshold:
                groups[-1].append(p)
            else:
                groups.append([p])

        return groups

    def _get_error_response(
        self, message: str, status: str = "error"
    ) -> Dict[str, Any]:
        """Створює стандартизовану відповідь з помилкою"""
        return {"status": status, "message": message}

    async def process_window(
        self,
        div_model: DivisionService.Model,
        gesture_model: GestureService.Model,
        stream: List[List[float]],
        detected_starts: List[int],
        detected_ends: List[int],
        is_end_request: bool,
    ) -> Tuple[
        Optional[Dict[str, Any]],
        List[List[float]],
        List[int],
        List[int],
        bool,
    ]:
        """
        Обробляє одне вікно даних та намагається розпізнати жест.

        Returns:
            (response, updated_stream, updated_starts, updated_ends, should_break)
        """
        left = len(stream) - DivisionService.WINDOW
        window_data = np.array(stream[-DivisionService.WINDOW :])

        detected_start, detected_end = await self.division_service.predict(
            div_model, window_data, left
        )

        if detected_start is not None:
            detected_starts.append(detected_start)

        if detected_end is not None:
            detected_ends.append(detected_end)

        detected_starts = sorted(detected_starts)
        detected_ends = sorted(detected_ends)

        start_groups = self.group_points(detected_starts, self.close_points_threshold)
        end_groups = self.group_points(detected_ends, self.close_points_threshold)

        print(
            {
                "start_groups": start_groups,
                "end_groups": end_groups,
            }
        )

        is_two_groups_detected = len(start_groups) >= 2 and len(end_groups) >= 2

        # Якщо не виявлено достатньо груп
        if not is_two_groups_detected and not is_end_request:
            return None, stream, detected_starts, detected_ends, False

        # Якщо это конець потоку та немає достатньо груп
        if not is_two_groups_detected and is_end_request:
            return await self._handle_end_request_single_gesture(
                start_groups,
                end_groups,
                stream,
                detected_starts,
                detected_ends,
                gesture_model,
            )

        # Якщо виявлено 2+ групи та це конець потоку
        if is_two_groups_detected and is_end_request:
            return await self._handle_end_request_multiple_gestures(
                start_groups,
                end_groups,
                stream,
                detected_starts,
                detected_ends,
                gesture_model,
            )

        # Якщо виявлено 2+ групи, але потік продовжується
        return await self._handle_streaming_detection(
            start_groups,
            end_groups,
            stream,
            detected_starts,
            detected_ends,
            gesture_model,
        )

    async def _handle_end_request_single_gesture(
        self,
        start_groups: List[List[int]],
        end_groups: List[List[int]],
        stream: List[List[float]],
        detected_starts: List[int],
        detected_ends: List[int],
        gesture_model: GestureService.Model,
    ) -> Tuple[Optional[Dict[str, Any]], List[List[float]], List[int], List[int], bool]:
        """Обробляє конець потоку з однією можливою групою жеста"""
        if len(start_groups) == 0 or len(end_groups) == 0:
            return (
                self._get_error_response("Not enough data to detect a gesture"),
                stream,
                detected_starts,
                detected_ends,
                True,
            )

        start_pt = int(np.mean(start_groups[0]))
        end_pts = [int(np.mean(g)) for g in end_groups]

        potential_end_pts = [e for e in end_pts if e > start_pt]

        if not potential_end_pts:
            return (
                self._get_error_response("Not enough data to detect a gesture"),
                stream,
                detected_starts,
                detected_ends,
                True,
            )

        end_pt = potential_end_pts[0]

        if end_pt - start_pt >= self.min_gesture_length:
            response = await self._recognize_gesture(
                stream, start_pt, end_pt, gesture_model
            )
            response["bounds"] = [start_pt, end_pt]
            response["status"] = "recognized"
            return response, stream, detected_starts, detected_ends, True

        # Спробувати наступні точки
        for end_pt in potential_end_pts[1:]:
            if end_pt - start_pt >= self.min_gesture_length:
                response = await self._recognize_gesture(
                    stream, start_pt, end_pt, gesture_model
                )
                response["bounds"] = [start_pt, end_pt]
                response["status"] = "recognized"
                return response, stream, detected_starts, detected_ends, True

        return (
            self._get_error_response("Gesture data too short"),
            stream,
            detected_starts,
            detected_ends,
            True,
        )

    async def _handle_end_request_multiple_gestures(
        self,
        start_groups: List[List[int]],
        end_groups: List[List[int]],
        stream: List[List[float]],
        detected_starts: List[int],
        detected_ends: List[int],
        gesture_model: GestureService.Model,
    ) -> Tuple[Optional[Dict[str, Any]], List[List[float]], List[int], List[int], bool]:
        """Обробляє конець потоку з декількома групами жестів"""
        start_pts = [int(np.mean(g)) for g in start_groups]
        end_pts = [int(np.mean(g)) for g in end_groups]

        # Обробляємо перший жест
        i = 0
        while i < len(start_pts):
            start_pt = start_pts[i]
            potential_end_pts = [e for e in end_pts if e > start_pt]

            if not potential_end_pts:
                return (
                    self._get_error_response("Not enough data to detect a gesture"),
                    stream,
                    detected_starts,
                    detected_ends,
                    True,
                )

            end_pt = potential_end_pts[0]

            if end_pt - start_pt >= self.min_gesture_length:
                response = await self._recognize_gesture(
                    stream, start_pt, end_pt, gesture_model
                )
                response["bounds"] = [start_pt, end_pt]
                response["status"] = "recognized"
                return response, stream, detected_starts, detected_ends, True

            is_start_small = len(start_pts) <= i + 1
            is_end_small = len(potential_end_pts) < 2

            if (is_start_small and is_end_small) or (
                not is_start_small and is_end_small
            ):
                return (
                    self._get_error_response("Not enough data to detect a gesture"),
                    stream,
                    detected_starts,
                    detected_ends,
                    True,
                )
            elif is_start_small and not is_end_small:
                if potential_end_pts[1] - start_pt >= self.min_gesture_length:
                    response = await self._recognize_gesture(
                        stream, start_pt, potential_end_pts[1], gesture_model
                    )
                    response["bounds"] = [start_pt, potential_end_pts[1]]
                    response["status"] = "recognized"
                    return response, stream, detected_starts, detected_ends, True
                else:
                    index = end_pts.index(potential_end_pts[1])
                    end_pts = end_pts[index + 1 :]
            else:
                if potential_end_pts[1] - start_pts[i + 1] >= self.min_gesture_length:
                    i += 1
                else:
                    end_pts = end_pts[1:]

            i += 1

        return (
            self._get_error_response("Not enough data to detect a gesture"),
            stream,
            detected_starts,
            detected_ends,
            True,
        )

    async def _handle_streaming_detection(
        self,
        start_groups: List[List[int]],
        end_groups: List[List[int]],
        stream: List[List[float]],
        detected_starts: List[int],
        detected_ends: List[int],
        gesture_model: GestureService.Model,
    ) -> Tuple[Optional[Dict[str, Any]], List[List[float]], List[int], List[int], bool]:
        """Обробляє потік з виявленими групами жестів"""
        start_pt = int(np.mean(start_groups[0]))
        end_pts = [int(np.mean(g)) for g in end_groups]

        potential_end_pts = [e for e in end_pts if e > start_pt]

        if not potential_end_pts:
            if len(end_pts) > 0:
                detected_ends = []
            return None, stream, detected_starts, detected_ends, False

        end_pt = potential_end_pts[0]

        if end_pt - start_pt >= self.min_gesture_length:
            response = await self._recognize_gesture(
                stream, start_pt, end_pt, gesture_model
            )
            response["bounds"] = [start_pt, end_pt]
            response["status"] = "recognized"

            # Очистимо потік
            stream, detected_starts, detected_ends = self._cleanup_stream(
                stream, detected_starts, detected_ends, start_groups, end_groups, end_pt
            )

            return response, stream, detected_starts, detected_ends, False

        elif len(potential_end_pts) >= 2:
            next_start_pt = int(np.mean(start_groups[1]))

            if potential_end_pts[1] < next_start_pt:
                end_pt_index = end_pts.index(end_pt)
                end_pts_to_remove = sum(
                    [len(group) for group in end_groups[: end_pt_index + 1]]
                )
                detected_ends = detected_ends[end_pts_to_remove:]
                return None, stream, detected_starts, detected_ends, False

        # Очистимо потік
        stream, detected_starts, detected_ends = self._cleanup_stream(
            stream, detected_starts, detected_ends, start_groups, end_groups, end_pt
        )

        return None, stream, detected_starts, detected_ends, False

    async def _recognize_gesture(
        self,
        stream: List[List[float]],
        start_pt: int,
        end_pt: int,
        gesture_model: GestureService.Model,
    ) -> Dict[str, Any]:
        """
        Розпізнає жест на основі даних потоку

        Args:
            stream: Потік даних
            start_pt: Індекс початку жеста
            end_pt: Індекс кінця жеста
            gesture_model: Модель розпізнавання жестів

        Returns:
            Словник з передбаченим жестом та впевненістю
        """
        gesture_data = stream[start_pt : end_pt + 1]
        response = await self.gesture_service.predict(gesture_model, gesture_data)
        return response

    def _cleanup_stream(
        self,
        stream: List[List[float]],
        detected_starts: List[int],
        detected_ends: List[int],
        start_groups: List[List[int]],
        end_groups: List[List[int]],
        end_pt: int,
    ) -> Tuple[List[List[float]], List[int], List[int]]:
        """
        Очищує потік від оброблених даних.

        Returns:
            (updated_stream, updated_starts, updated_ends)
        """
        shift = end_pt + 1
        stream = stream[shift:]
        detected_starts = detected_starts[len(start_groups[0]) :]
        end_pt_index = [int(np.mean(g)) for g in end_groups].index(
            int(np.mean(end_groups[0]))
        )
        end_pts_to_remove = sum(
            [len(group) for group in end_groups[: end_pt_index + 1]]
        )
        detected_ends = detected_ends[end_pts_to_remove:]
        detected_starts = [p - shift for p in detected_starts if p - shift >= 0]
        detected_ends = [p - shift for p in detected_ends]

        return stream, detected_starts, detected_ends
