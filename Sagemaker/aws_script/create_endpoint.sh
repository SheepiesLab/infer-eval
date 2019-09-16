#!/usr/bin/env bash

MODEL_NAME=sm-keras

ENDPOINT_CONFIG_NAME=sm-keras-1

ENDPOINT_NAME=sm-keras

PRODUCTION_VARIANTS="VariantName=Default,ModelName=${MODEL_NAME},"\
"InitialInstanceCount=1,InstanceType=ml.m5.large"

aws sagemaker create-endpoint-config --endpoint-config-name ${ENDPOINT_CONFIG_NAME} \
--production-variants ${PRODUCTION_VARIANTS}

aws sagemaker create-endpoint --endpoint-name ${ENDPOINT_NAME} \
--endpoint-config-name ${ENDPOINT_CONFIG_NAME}