from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy import stats as st
from io import BytesIO
from PIL import Image

import numpy as np
import logging
import base64
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

import azure.functions as func

# Initialize models
classifier_1 = load_model("./nephro_ai/classifier_models/ct_classifier_1.h5")
classifier_2 = load_model("./nephro_ai/classifier_models/ct_classifier_2.h5")
classifier_3 = load_model("./nephro_ai/classifier_models/ct_classifier_3.h5")

model_1 = load_model('./nephro_ai/models/R1_model_tumor.h5')
model_2 = load_model('./nephro_ai/models/R2_model_tumor.h5')
model_3 = load_model('./nephro_ai/models/R3_model_tumor.h5')
model_4 = load_model('./nephro_ai/models/R4_model_stone.h5')
model_5 = load_model('./nephro_ai/models/R5_model_stone.h5')
model_6 = load_model('./nephro_ai/models/R6_model_stone.h5')
model_7 = load_model('./nephro_ai/models/R9_model_cyst.h5')
model_8 = load_model('./nephro_ai/models/R10_model_cyst.h5')
model_9 = load_model('./nephro_ai/models/R11_model_cyst.h5')

# Make a list of models
models = [classifier_1, classifier_2, classifier_3, model_1, model_2, 
          model_3, model_4, model_5, model_6, model_7, model_8, model_9]
    
# Disable further training of models
for model in models:
    model.trainable = False
    
    
# Determine the file type
def get_image_type_from_base64(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        
        return image.format.lower()
    
    except:
        raise AttributeError("Invalid file type!") 

# Decode and convert the image in to np array
def base64_to_array(base64_string: str,
                    image_size: tuple
                    ) -> np.ndarray:
    try:
        logging.info('Decoding the image from base64 to numpy.')
        decoded_base64 = base64.b64decode(base64_string)
        
        try:
            logging.info(f'Resizing the image to size {image_size}.')
            img = load_img(BytesIO(decoded_base64), target_size=image_size)
            
            # Convert the image to a numpy array
            img = img_to_array(img) / 255.0

            # Reshape the image to add a batch dimension
            img = np.expand_dims(img, axis=0)
            
            logging.info('Resizing has been completed.')
            return img
        
        except:
            raise UnicodeDecodeError("Poor image quality!")
        
    except UnicodeDecodeError as ue:
        raise UnicodeDecodeError(ue)
    
    except Exception:
        raise RuntimeError("Invalid image found!")

def verify(img):
    try:

        # Predict using the generator
        predictions = []
        for model in models[:3]:
            prediction = model.predict(img, verbose=0)
            predictions.append(prediction)

        normalised_result = []
        for i in predictions:
            normalised_result.append(np.argmax(i, axis=1))

        final_prediction = st.mode(normalised_result, keepdims=False).mode[0]
        result = classify(final_prediction)
        
        return result
    
    except:
        return "Invalid file type found!"
    
# Classify CT images
def classify(prediction):
    return 'CT' if prediction == 0 else 'Non-CT'
    
# Classift the predictions
def prediction_class(prediction, class_no):
    if class_no == 1 or class_no == 2:
        return 'Negative' if prediction == 0 else 'Positive'
    else:
        return 'Positive' if prediction == 0 else 'Negative'
    
def predict(img):
    try:
        # Predict using the generator
        logging.info('Starting the prediction process.')
        
        predictions = []
        for i, model in enumerate(models[3:]):
            
            logging.info(f'Sending the image to model - {i + 1}')
            prediction = model.predict(img, verbose=0)
            predictions.append(prediction)

        # Normalise predictions
        logging.info('Normalising the predicted classes')
        
        npx_prediction = []
        for prediction in predictions:
            npx_prediction.append(np.argmax(prediction, axis=1))

        logging.info('Classifying the class models')
        npx_tumor = st.mode(npx_prediction[:3], keepdims=False)
        npx_stone = st.mode(npx_prediction[3:6], keepdims=False)
        npx_cyst = st.mode(npx_prediction[6:], keepdims=False)

        # Classify the final predictions
        result_1 = prediction_class(npx_tumor.mode[0], 1)
        result_2 = prediction_class(npx_stone.mode[0], 2)
        result_3 = prediction_class(npx_cyst.mode[0], 3)

        # Calculate the confidence score
        logging.info('Calculating the confidence score.')
        npx_tm = sum([round(max(score[0]) * 100, 2) for score in predictions[:3]])
        npx_st = sum([round(max(score[0]) * 100, 2) for score in predictions[3:6]])
        npx_cy = sum([round(max(score[0]) * 100, 2) for score in predictions[6:]])

        score_1 = str(round(npx_tm / 3, 2)) + '%'
        score_2 = str(round(npx_st / 3, 2)) + '%'
        score_3 = str(round(npx_cy / 3, 2)) + '%'

        return [result_1, result_2, result_3, score_1, score_2, score_3]
    
    except RuntimeError as re:
        raise RuntimeError(re)
    
    except Exception:
        raise RuntimeError("Error processing the input tensor")


# Main method of the program
def main(request: func.HttpRequest) -> func.HttpResponse:
    if request.method == "POST":
        logging.info('Received a POST request.')
        
        try:
            req_body = request.get_json()
            img = req_body.get('img')
        except ValueError:
            return func.HttpResponse("Invalid request. Please make sure that your request is valid!", 400)

        if img:
            try:
                logging.info('Processing the request.')
                logging.info('Starting verification.')
                file_type = get_image_type_from_base64(img)
                
                if file_type in ["jpeg", "jpg", "png"]:
                    
                    img_1 = base64_to_array(img, (224, 224))
                    
                    # Find the ct type
                    logging.info('Validating CT image type.')
                    
                    img_2 = base64_to_array(img, (64, 64))
                    ct_type = verify(img_2)
                    
                    if ct_type == 'CT':
                    
                        logging.info('Verification completed successfully.')
                    
                        logging.info('Attempting to predict from the input tensors.')
                        predictions = predict(img_1)
                        
                        logging.info('Process has been completed successfully.')
                        return func.HttpResponse(f'{predictions}', status_code=200)
                    
                    else:
                        raise TypeError("Not a CT scan image!")
                else:
                    raise ValueError(f"Unsupported file format found: {file_type}")
                
            except RuntimeError:
                logging.info('Process terminated due to an error while processing tensor.')
                return func.HttpResponse("Invalid image found!", status_code=400)
            
            except TypeError:
                logging.info('Process terminated due a non-ct scan image.')
                return func.HttpResponse("Invalid CT scan image found!", status_code=400)
                
            except UnicodeDecodeError as e:
                logging.info('Process terminated due to poor image quality.')
                return func.HttpResponse("Poor image quality!", status_code=400)
                
            except (ValueError, AttributeError):
                logging.info(f'Process terminated due to an invalid file type.')
                return func.HttpResponse("Unsupported file format!", status_code=400)
                
            except Exception as e:
                logging.info(f'Process terminated due to an unknown error. Complete log is as follows, {str(e)}')
                return func.HttpResponse("The application is under maintenance. Please try again later!", status_code=500)
        else:
            logging.info('Received a POST request with an empty body.')
            return func.HttpResponse("File does not exists!", status_code=404)
    else:
        return func.HttpResponse("The server is up and running.", status_code=200)