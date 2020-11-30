import flask
import joblib
import os
import glob
from fastai.vision.all import *
import pandas as pd

app = flask.Flask(__name__)#, template_folder='templates')

# Config settings
app.config["IMAGE_UPLOADS"] = "static/img"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "JFIF"]
app.config["MAX_IMAGE_FILESIZE"] = 250000

# # Setup for model import
# bs = 64 # Batch size
# resize_size = 180 # for training, resize all the images to a square of this size
# training_subsample = 0.8 # for development, use a small fraction of the entire dataset rater than full dataset
# df_labels = pd.read_csv('labels.csv')
# df_labels=df_labels.set_index('id')
# df_labels = df_labels.sample(frac=training_subsample, axis=0)
# data = ImageDataLoaders.from_df(
#     df = df_labels,
#     # path = ('C:/Users/JBStogner/GitHub/Project-3/kaggle_bee_vs_wasp'),
#     valid_pct=0.2,
#     seed = 42,
#     fn_col='path',
#     folder=None,
#     label_col='label',
#     bs=bs,
#     shuffle_train=True,
#     batch_tfms=aug_transforms(),
#     item_tfms=Resize(resize_size),device='cpu', num_workers=0,
# )

# learn = cnn_learner(data, arch=models.resnet34, path='model.pth', metrics=[error_rate, accuracy])
# learn = cnn_learner(data, models.resnet34, metrics=[error_rate, accuracy], pretrained=True)
# learn.load('model')
learn = load_learner('models/model')

# Function to check file extension (imgrecognition)
def allowed_image(filename):

    if not "." in filename:
        return False

    global ext
    ext = filename.rsplit(".", 1)[1]
    
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

@app.route('/', methods=['GET', 'POST'])

def img_predict():

    # Run on Submit button click
    if flask.request.method == 'POST':

        # learn = cnn_learner(data, models.resnet34, metrics=[error_rate, accuracy])

        # This works, but is clumsy - can't figure out a better way to do it
        # Without this the file caches and doesn't update on reload
        # Identify the file to be deleted
        del_file = glob.glob(app.config["IMAGE_UPLOADS"] + '/UPLOAD_PIC*')
        # Convert to a string and remove the root
        file_to_delete = str(del_file)[14:-2]

        try:
            # Delete the previously uploaded file
            os.remove(app.config["IMAGE_UPLOADS"] + '/' + file_to_delete)
        
        except:
            print('')

        # Get the file name
        image = flask.request.files['image']

        # Get the image file size
        image.seek(0, os.SEEK_END)
        size = image.tell()
        
        # If file size > size limit in config settings then do not accept the image
        if size > app.config["MAX_IMAGE_FILESIZE"]:
            return(flask.render_template('imgrecognition.html', prediction="Maximum file size exceeded."))
        
        if allowed_image(image.filename):

            now = datetime.now()
            substr_now = str(now)[-6:]

            # Use a constant filename, but with a variable extension - this facilitates deletion later
            filename = 'UPLOAD_PIC' + substr_now + '.' + ext
            
            # fastai creates an image object - not really sure why this is necessary
            img = PILImage.create(image)

            # Save the image
            img.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
    
            # Load the model
            # learn.load('model')
                    
            # Run the image through the model
            pred_class, pred_idx, outputs = learn.predict(img)

            # Return the prediction on the webpage and display the image
            return(flask.render_template('imgrecognition.html', prediction=f'Prediction class: {pred_class}', selected_image=app.config["IMAGE_UPLOADS"] + '/' + filename))
            
        else:
            return(flask.render_template('imgrecognition.html', prediction="Please select a valid file type."))
    
    if flask.request.method == 'GET':
        
        return(flask.render_template('imgrecognition.html'))

if __name__ == '__main__':
    app.run()
