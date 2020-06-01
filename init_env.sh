#!/bin/bash

# check if storage path is specified
if [ "$#" -ne 2 ]; then
  echo "Illegal number of arguments. Please specify project name and storage drive path"
  exit 1
fi

PROJECT=$1
echo "PROJECT=${PROJECT}"

STORAGE=$2
echo "STORAGE=${STORAGE}"

PROJECT_STORAGE_ROOT="${STORAGE}/${PROJECT}"
echo "PROJECT_STORAGE_ROOT=${PROJECT_STORAGE_ROOT}"

SRC_ROOT=${PWD}
echo "SRC_ROOT=${SRC_ROOT}"

DATASETS="${PROJECT_STORAGE_ROOT}/datasets"
echo "DATASETS=${DATASETS}"

MODELS="${PROJECT_STORAGE_ROOT}/models"
echo "MODELS=${MODELS}"

LOG="${PROJECT_STORAGE_ROOT}/log"
echo "LOG=${LOG}"

# create storage path in case it doesn't exists
mkdir -p "${PROJECT_STORAGE_ROOT}"
mkdir -p "${DATASETS}"
mkdir -p "${MODELS}"
mkdir -p "${LOG}"

# create .env file in project root and add vars to it
ENV="${SRC_ROOT}/.env"
echo "SRC_ROOT=${PWD}
PROJECT=${PROJECT}
STORAGE=${STORAGE}
PROJECT_STORAGE_ROOT=${PROJECT_STORAGE_ROOT}
DATASETS=${DATASETS}
MODELS=${MODELS}
LOG=${LOG}
PYTHONPATH=\${PYTHONPATH}:\${SRC_ROOT}" > "${ENV}"
