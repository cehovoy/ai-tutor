#!/bin/bash

echo "Остановка контейнеров AI-репетитора..."
docker-compose down

echo -e "\nСписок остановленных контейнеров:"
docker-compose ps

echo -e "\nГотово! Контейнеры остановлены." 