from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from scipy import stats as st
from io import BytesIO
from PIL import Image, ImageOps
from skimage.transform import resize
from skimage.io import imread

import numpy as np
import logging
import base64
import tensorflow as tf
import azure.functions as func

tf.get_logger().setLevel('ERROR')


# Initialize models
model_0 = load_model("./nephro_ai/models/CT_CLF.h5", compile=False)
model_1 = load_model('./nephro_ai/models/R1_TMR_01.h5')
model_2 = load_model('./nephro_ai/models/R2_STN_01.h5')
model_3 = load_model('./nephro_ai/models/R2_STN_02.h5')
model_4 = load_model('./nephro_ai/models/R3_CST_01.h5')

# Make a list of models
models = [model_0, model_1, model_2, model_3, model_4]

# Disable further training of models
for model in models:
    model.trainable = False

# Declare custom exceptions
class InvalidFileTypeError(Exception):
    pass
class FileSizeTooSmallError(Exception):
    pass
class FileSizeExceededError(Exception):
    pass
class FileNormalizationError(Exception):
    pass
class Base64DecodeError(Exception):
    pass

# Determine the file type
def get_image_type_from_base64(base64_string: str
    ) -> str:
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        return image.format.lower()
    
    except:
        raise InvalidFileTypeError("Unsupported file format found!") 
    
# Initialise the pre-processor
def normalized_data(img: np.ndarray
    ) -> np.ndarray:
    try:
        img -= np.mean(img, axis=(0,1))
        img /= (np.std(img, axis=(0,1)) + 1e-7)
        return img 
    
    except:
        raise FileNormalizationError("Poor image quality!")
    
# Classift the predictions
def prediction_class(prediction: int
    ) -> str:
    return 'Negative' if prediction == 0 else 'Positive'

# Process the data for stone prediction
def supervised_preprocessor(path: str
    ) -> np.ndarray:
    try:
        X = np.zeros((1, 128, 128, 3), dtype=np.uint8)
        img = imread(BytesIO(path))
        img = resize(img, (128, 128, 3), 
                        mode='constant', preserve_range=True)
        X[0] = img
        return X
    
    except Exception:
        raise FileNormalizationError("Poor image quality!")

# Decode and convert the image in to np array
def decode_base64(base64_string: str
    ) -> str:
    try:
        logging.info('Decoding the image from base64 to numpy.')
        decoded_base64 = base64.b64decode(base64_string)
        return decoded_base64

    except Exception:
        raise Base64DecodeError("Invalid image found!")

# Verify the image
def verify(
    decoded_base64: str, 
    encoded_base64: str
    ) -> str:

    # Load image buffer
    image_bytes = BytesIO(decoded_base64)
        
    # Check the base64 length
    logging.info('Checking image size.')
    BYTE_CODE = 1.0
    if encoded_base64[-2:] == '==': 
        BYTE_CODE = 2.0
        
    # Calculate file size in bytes
    size_bytes = (len(encoded_base64) * (3.0/4.0)) - BYTE_CODE
    
    # Convert to megabytes and verify the file size
    size_mb = size_bytes * 1e-6
    
    # Validate the file size
    if size_mb > 5.0: # Should not exceed 5MB
        raise InvalidFileTypeError("Image size exceeded the limit!")
    
    elif size_mb < 0.080: # Should not less than 80KB
        raise FileSizeTooSmallError("Image size is too small!")
    
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Load the image
    image = Image.open(image_bytes).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize and load the image into the array
    data[0] = (image_array.astype(np.float32) / 127.5) - 1

    # Predicts the model
    logging.info('Validating CT image class.')
    prediction = models[0].predict(data, verbose=0)
    index = np.argmax(prediction)
    
    # Classify from the predicted class
    result = prediction_class(index)    
        
    return result 


# Preprocess multiple image
def image_preprocessor(
    decoded_base64: str, 
    divide: bool
    ) -> ImageDataGenerator:
    
    # Load the image
    img = load_img(BytesIO(decoded_base64), target_size=(224, 224))

    # Convert the image to a numpy array
    img = img_to_array(img)

    # Divide by 255.0 to normalise data
    if divide: 
        img = img / 255.0
        
    # Reshape the image to add a batch dimension
    img = np.expand_dims(img, axis=0)

    # Normalise the image
    dataGen = ImageDataGenerator(
        horizontal_flip=True, 
        vertical_flip=True,
        rotation_range=90,
        preprocessing_function=lambda img: normalized_data(img))
    
    normalised_image = dataGen.flow(img)
    
    return normalised_image


# Predict diseases using classifiers
def predict(img: str
    ) -> list:

    # Predict using the generator
    logging.info('Starting the prediction process.')
    
    img_1 = image_preprocessor(img, False)
    img_2 = image_preprocessor(img, True)
    img_3 = supervised_preprocessor(img)
    
    # Predict based on the preprocessing technique
    predictions = []
    for i, model in enumerate(models[1:]):
        
        logging.info(f'Sending the image to model - {i + 1}')
        if i in [0, 3]:
            prediction = model.predict(img_1, verbose=0)
        elif i == 1:
            prediction = model.predict(img_2, verbose=0)
        else:
            prediction = model.predict(img_3, verbose=0)[0]
        predictions.append([prediction])
            
    logging.info('Normalising the predicted classes')
    
    # Classify Tumor
    tumor_class = np.argmax(predictions[0])

    # Classify Stone
    stone_classes = []
    for i, class_x in enumerate(predictions[1:3]):
        if i == 1: # Predict using the supervised learning model
            class_x = (class_x[0] > 0.5).astype(np.uint8)
        max_class = np.argmax(class_x)
        stone_classes.append(max_class)
    stone_class = max(stone_classes)

    # Classify Cyst
    cyst_classes = np.argmax(predictions[-1])

    # Classify the final results
    result_1 = prediction_class(tumor_class)
    result_2 = prediction_class(stone_class)
    result_3 = prediction_class(cyst_classes)
    
    return [result_1, result_2, result_3]


# Main method of the program
def main(request: func.HttpRequest
    ) -> func.HttpResponse:
    
    if request.method == "POST":
        logging.info('Received a POST request.')        
        try:
            req_body = request.get_json()
            img = req_body.get('img')
            
            if img == '':
                raise ValueError
        except ValueError:
            logging.info('Got an empty body. The process has been terminated.')
            return func.HttpResponse("Empty body, please make sure that your request is valid!", status_code=404)
        except Exception:
            logging.info('Got an invalid request. The process has been terminated.')
            return func.HttpResponse("Invalid request, please make sure that your request is valid!", status_code=406)
        
        if img:
            try:
                logging.info('Starting verification.')
                file_type = get_image_type_from_base64(img)
                
                if file_type in ["jpeg", "jpg", "png"]:
                    logging.info('Validating CT image type.')
                    img_decoded = decode_base64(img)
                    ct_type = verify(img_decoded, img)
                    
                    if ct_type == 'Negative':
                        logging.info('Attempting to predict from the input tensors.')
                        predictions = predict(img_decoded)
                        
                        logging.info('Process has been completed successfully.')
                        return func.HttpResponse(f'{predictions}', status_code=200)
                    else:
                        raise InvalidFileTypeError("The image did not meet CT standards!")
                else:
                    raise InvalidFileTypeError("Unsupported file format found!")
                
            except InvalidFileTypeError as IFTe:
                logging.info('Process terminated due to an invalid file format.')
                return func.HttpResponse(f'{str(IFTe)}', status_code=400)
            
            except FileSizeTooSmallError as FTTSe:
                logging.info('Process terminated due to file size is too small.')
                return func.HttpResponse(f'{str(FTTSe)}', status_code=400)
            
            except FileSizeExceededError as FSEe:
                logging.info('Process terminated due to file size is too large.')
                return func.HttpResponse(f'{str(FSEe)}', status_code=400)
                
            except FileNormalizationError as FNe:
                logging.info('Process terminated due to poor image quality.')
                return func.HttpResponse(f'{str(FNe)}', status_code=400)
                
            except Base64DecodeError as BSTe:
                logging.info('Process terminated due to an invalid file type.')
                return func.HttpResponse(f'{str(BSTe)}', status_code=400)
                
            except Exception as e:
                logging.info(f'Process terminated due to {str(e)}')
                return func.HttpResponse("The application is under maintenance. Please try again later!", status_code=500)
        else:
            logging.info('Received a POST request with an empty body.')
            return func.HttpResponse("File does not exists!", status_code=404)
    else:
        return func.HttpResponse("The server is up and running.", status_code=200)