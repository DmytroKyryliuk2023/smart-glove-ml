if [[ "$PWD" == *start_server ]]; then
  cd ..
fi

uvicorn app.main:app --reload