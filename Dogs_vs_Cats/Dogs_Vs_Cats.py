
# coding: utf-8

# In[2]:


from fastai.vision import *
import zipfile


# In[3]:


#!pip install kaggle                # Installs Kaggle API to download package and configure environment.
#! mkdir -p ~/.kaggle/
#! mv kaggle.json ~/.kaggle/
#!kaggle competitions download -c dogs-vs-cats      # Command that actually downloads files


# In[4]:


def unzip(path):
    
    # Function Description: Given a path name to a folder, unzip the folder.
    # Function Parameters: path (The path to the folder)
    # Function Throws / Returns: Nothing
    
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall('.')
        
unzip('dogs-vs-cats.zip')         # Unpack the test and training data.
unzip('train.zip')


# In[5]:


path_img = Path('train')             # Read the names of all the picture files.
fnames = get_image_files(path_img)
print(fnames[-2:])      # Dogs
print(fnames[:2])       # Cats


# In[6]:


np.random.seed(2)
reg = r'/([^/]+)\.\d+.jpg$'


# In[7]:


data = ImageDataBunch.from_name_re(path_img, fnames, pat=reg, ds_tfms=get_transforms(), size=224)
data.normalize(imagenet_stats)


# In[8]:


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


# In[9]:


# Setting up data for Resnet50 - Use a smaller bs if you run out of memory (bs)
data = ImageDataBunch.from_name_re(path_img, fnames, pat=reg, ds_tfms=get_transforms(), size=224)
data.normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(5,7))
print(data.classes),data.c


# In[10]:


# Attempt to use Resnet50
learn = cnn_learner(data, models.resnet50, metrics=error_rate)


# In[11]:


learn.fit_one_cycle(5)


# In[13]:


learn.save("res50")
learn.lr_find()
learn.recorder.plot()


# In[14]:


learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-3))


# In[ ]:




