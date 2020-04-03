
# coding: utf-8

# In[1]:


from fastai.vision import *
import zipfile


# In[2]:


#!pip install kaggle                # Installs Kaggle API to download package and configure environment.
#! mkdir -p ~/.kaggle/
#! mv kaggle.json ~/.kaggle/
#!kaggle competitions download -c dogs-vs-cats      # Command that actually downloads files


# In[9]:


def unzip(path):
    
    # Function Description: Given a path name to a folder, unzip the folder.
    # Function Parameters: path (The path to the folder)
    # Function Throws / Returns: Nothing
    
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall('.')
        
unzip('dogs-vs-cats.zip')         # Unpack the test and training data.
unzip('train.zip')


# In[10]:


path_img = Path('train')             # Read the names of all the picture files.
fnames = get_image_files(path_img)
print(fnames[-2:])      # Dogs
print(fnames[:2])       # Cats


# In[11]:


np.random.seed(2)
reg = r'/([^/]+)\.\d+.jpg$'


# In[12]:


data = ImageDataBunch.from_name_re(path_img, fnames, pat=reg, ds_tfms=get_transforms(), size=224)
data.normalize(imagenet_stats)


# In[14]:


data.show_batch(rows=3, figsize=(5,7))
print(data.classes),data.c


# In[17]:


# Create a Convoluted Neural Network
learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[18]:


# Train the Neural Net
learn.fit_one_cycle(4)


# In[19]:


learn.save("first-net")


# In[21]:


interp = ClassificationInterpretation.from_learner(learn)


# In[23]:


interp.plot_top_losses(9, figsize=(11, 15))
# Using the line above we can plot the images that were classified incorrectly.
# It is evident that some of the pictures can use some enhancement (touch ups), 
# Or some sort of rescaleling. (Prediction, actual, loss, probability)


# In[34]:


learn.load("first-net")
learn.lr_find()    # Time to train the entire model.


# In[35]:


learn.recorder.plot()


# In[36]:


learn.unfreeze()        # Train the entire model with adjusted learning rate.
learn.fit_one_cycle(2, max_lr=slice(1e-7, 1e-5))

