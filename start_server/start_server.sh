if [[ "$PWD" != *start_server ]]; then
  cd app
fi

uvicorn main:app --reload