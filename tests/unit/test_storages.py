from unittest.mock import Mock, patch

import numpy as np
import pytest

from app.storages import ModelMinIOStorage


@pytest.mark.asyncio
async def test_save_model(mock_minio_client, sample_model):
    """Test saving model to MinIO"""
    storage = ModelMinIOStorage(mock_minio_client, "test-bucket")

    with patch("app.storages.tempfile.TemporaryDirectory") as mock_tempdir:
        mock_tempdir.return_value.__enter__.return_value = "/tmp/test"

        with patch("app.storages.tf_save_model"):
            with patch("app.storages.joblib.dump"):
                with patch("app.storages.np.save"):
                    with patch("os.path.exists", return_value=True):
                        await storage.save_model("test_id", sample_model)

                        assert mock_minio_client.fput_object.call_count == 3


@pytest.mark.asyncio
async def test_load_model(mock_minio_client):
    """Test loading model from MinIO"""
    storage = ModelMinIOStorage(mock_minio_client, "test-bucket")

    with patch("app.storages.tempfile.TemporaryDirectory") as mock_tempdir:
        mock_tempdir.return_value.__enter__.return_value = "/tmp/test"

        with patch("app.storages.tf_load_model") as mock_load:
            with patch("app.storages.joblib.load") as mock_joblib:
                with patch("app.storages.np.load") as mock_np:
                    mock_load.return_value = Mock()
                    mock_joblib.return_value = Mock()
                    mock_np.return_value = np.array(["a", "b"])

                    # Mock minio fget_object to avoid file operations
                    def mock_fget_object(bucket, name, file_path):
                        pass

                    mock_minio_client.fget_object = mock_fget_object

                    result = await storage.load_model("test_id")

                    assert result is not None


def test_bucket_creation_on_init():
    """Test bucket creation during initialization"""
    mock_client = Mock()
    mock_client.bucket_exists.return_value = False

    ModelMinIOStorage(mock_client, "new-bucket")

    mock_client.make_bucket.assert_called_once_with("new-bucket")


def test_bucket_exists_on_init():
    """Test when bucket already exists"""
    mock_client = Mock()
    mock_client.bucket_exists.return_value = True

    ModelMinIOStorage(mock_client, "existing-bucket")

    mock_client.make_bucket.assert_not_called()
