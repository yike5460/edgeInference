# overall process tuturial (rasp4, rk3288)

## use pretrained model
retrieve model, and try pretained model, refer to pretrained.py. note such model only contain params file.
```
wget -c https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/yolo3_mobilenet1.0_voc-3b47835a.zip
```

offcical site refer to https://cv.gluon.ai/model_zoo/detection.html#yolo-v3

export trained gluoncv network to json, refer to file exportNetwork.py

### export network json file
pretrained model only include params file not network json, make sure to export refer to https://cv.gluon.ai/build/examples_deployment/export_network.html


## train your own model
refer to https://cv.gluon.ai/build/examples_detection/train_yolo_v3.html#sphx-glr-build-examples-detection-train-yolo-v3-py


## use neo to compile crossplatform lib (rasp4 e.g.)
pack model file (params and json) and input to sagemaker compilation job, with **Data input configuration** set to {"data": [1,3,224,224]} according to sample.
```
tar -czvf yolo3_mobilenet.tar.gz yolo3_mobilenet1.0_voc-0000.params yolo3_mobilenet1.0_voc-symbol.json
```

code to compile neo with api
```
data_shape = '{"data": [1,416,416,3]}'
framework = 'mxnet'
target_device = 'rk3288'
role_arn = 'xxx'

import time
compilation_job_name = 'JupyterRK3388-'+ str(time.time()).split('.')[0]
print(f'Compilation job for {compilation_job_name} started')

response = sagemaker_client.create_compilation_job(
    CompilationJobName=compilation_job_name,
    RoleArn=role_arn,
    InputConfig={
        'S3Uri': s3_input_location,
        'DataInputConfig': data_shape,
        'Framework': framework.upper()
    },
    OutputConfig={
        'S3OutputLocation': s3_output_location,
        'TargetDevice': target_device,
         #'TargetPlatform': {"Os": "LINUX", "Arch": "ARM_EABIHF", "Accelerator": "MALI"}
    },
    StoppingCondition={
        'MaxRuntimeInSeconds': 900
    }
)
```

## export compiled files
download files from configured output s3 bucket, and unzip to working folder, files shoudl include:
- compiled.meta
- compiled_model.json
- compiled.params
- compiled.so
- dlr.h
- libdlr.so
- manifest

## install dlr
### prebuilt source
On x86_64 CPU targets running Linux, you can install latest release of DLR package via

```
pip install dlr
```

For installation of DLR on GPU targets or non-x86 edge devices, please refer to [Releases](https://github.com/neo-ai/neo-ai-dlr/releases) for prebuilt binaries, or Installing DLR for [building DLR from source](https://neo-ai-dlr.readthedocs.io/en/latest/install.html).

```
wget https://neo-ai-dlr-release.s3-us-west-2.amazonaws.com/v1.8.0/rasp4b/dlr-1.8.0-py3-none-any.whl
pip install dlr-1.8.0-py3-none-any.whl 
```
### make from source
refer to https://neo-ai-dlr.readthedocs.io/en/latest/install.html#building-for-cpu

open opencl option and modify -lpthread to -pthread in CMakeList.txt to make sure DLR compile successful

## start inference
### rasp4
refer to rasp4Inference.py
outputs should be like 
```
pi@raspberrypi:~/inference $ sudo python rasp4Inference.py 

 CALL HOME FEATURE ENABLED
                            

 You acknowledge and agree that DLR collects the following metrics to help improve its performance.                             
 By default, Amazon will collect and store the following information from your device:                             

 record_type: <enum, internal record status, such as model_loaded, model_>,                             
 arch: <string, platform architecture, eg 64bit>,                             
 osname: <string, platform os name, eg. Linux>,                             
 uuid: <string, one-way non-identifable hashed mac address, eg. 8fb35b79f7c7aa2f86afbcb231b1ba6e>,                             
 dist: <string, distribution of os, eg. Ubuntu 16.04 xenial>,                             
 machine: <string, retuns the machine type, eg. x86_64 or i386>,                             
 model: <string, one-way non-identifable hashed model name, eg. 36f613e00f707dbe53a64b1d9625ae7d>                             

 If you wish to opt-out of this data collection feature, please follow the steps below:                             
        1. Disable it with through code:                             
                 from dlr.counter.phone_home import PhoneHome                             
                 PhoneHome.disable_feature()                            
        2. Or, create a config file, ccm_config.json inside your DLR target directory path, i.e. python3.6/site-packages/dlr/counter/ccm_config.json. Then added below format content in it, {"enable_phone_home" : false}                             
        3. Restart DLR application.                             
        4. Validate this feature is disabled by verifying this notification is no longer displayed, or programmatically with following command:                             
                from dlr.counter.phone_home import PhoneHome                             
                PhoneHome.is_enabled() # false as disabled 
2021-04-14 09:14:55,366 INFO Found libdlr.so in model artifact. Using dlr from /home/pi/inference/yolo3_mobilenet-rasp3b/libdlr.so
(1, 3, 224, 224)
Testing inference...
inference time is 2.412400484085083 seconds
1
Result: label -  'goldfish, Carassius auratus', probability - 17.0
```

### rk3388
error need to resolve, refer to rk3288Inference.py

# reference link
## pretained model and example code
https://cv.gluon.ai/model_zoo/detection.html#yolo-v3
https://cv.gluon.ai/build/examples_detection/demo_yolo.html#sphx-glr-build-examples-detection-demo-yolo-py

## TVM example (neo backend)
https://tvm.apache.org/docs/tutorials/frontend/from_mxnet.html

## raspberry pi example
https://github.com/neo-ai/neo-ai-dlr/tree/main/sagemaker-neo-notebooks/edge/raspberry-pi
