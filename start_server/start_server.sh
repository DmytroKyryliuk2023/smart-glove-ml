if [[ "$PWD" != *start_server ]]; then
  cd start_server
fi

uvicorn app.main:app --reload