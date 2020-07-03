# Simplest Image Classifier - Tutorial

This little image classifier is a showcase of how easy it is to create a functional image classifier using just basic python and some easy to use and well-documented librariees. I prepared this script for my Informatics I - Python Programmig tutorial session in November to show my students what some basic python programming can achieve! 


# Our Goal 
We want to create a simple python script that recognizes a hand-written number. The input image shall all have a fixed size of 64x64 pixels, the images are RGB coloured. Let's write a python script that analyzes the image and gives us a prediction what number is written on it. We will take the following steps to achieve this goal

  - First, we draw by hand a set of images that display every possible digit that our script shall recognize. This set of images will serve our script as learning data. 
  - Next, we want to write a method that opens and normalizes our image so that only the relevant information is kept. Colour and other distracting noise shall be removed completely.
  - Then, let's create a Database containing all well-defined learning images. We will compare a new input image to the images in the database to make an estimate which number it displays. 
  - Finally, we write the comparing algorithm. We take a new input image, normalize it and compare it to every image in our database wich computing the similarity based on pixel values. The image-category with the highest similarity will serve us as an estimate. 

We want to base our algorithm on training, meaning that we do not hardcode how it shall recognize the digit "1", but it shall recognize it by himself. We do this by providing him a set of well-defined training images, so images that are already classified, named according to the digit they display. Our algorithm will read the images and the class they belong to, and store it in a database. When a new, unknown image is provided, the algorithm loops through the database and computes the similarity to the well-known images. Let's break that down into smaller pieces and explain them in detail. 

### 

## Let's write some code!
Okay, now that we have an idea what our script shall do, lets translate them into actual python code.

### The needed Libraries
To open Images, we will use the Image class from the PIL (Python Image Library). To perform calculations on the image efficiently, we will use the scientific mathematic library numpy. Finally, to store our image database and open it later on, we will use pickle. While pickle is a standard library, the two others have to be installed using pip.

```sh
$ pip install numpy
$ pip install PIL
```

Let's import these libraries now. 

```Python
from PIL import Image
import numpy as np
import pickle
```

### The class structure
We want our image classifier to be a class. This has several advantages, for example, it makes it easy to import it to another project later on, and it gives us a well structured project in OOP style. If you don't know why to use classes, go and have a look at my other Github repo [Functional vs OOP Showcase](https://github.com/joelbarmettlerUZH/Functional_vs_OOP_Showcase)!

Our class shall have a **constructor** and a **str**-method, as well as a method to **create** & **open** a image-database, **normalize** an image, normalize a whole database and **classify** an image. 

The **constructor** shall take a *databaseName* with which we classify the image. The **createDatabase** method takes a *folder* in which the images are stored, and a *name of a database* that will be created containing representatios of these images. The **normalize** method takes an *image* that is to normalize. The **normalizeDatabase** method takes a *normalizationFunction*, just in case that we implement different normallization function later (then we can pass the one we want to apply on the DB as an argument into the method. If you are not familiar with this concept, read this article [here](https://dbader.org/blog/python-first-class-functions)).Pretty straight forward. Our empty class schema will look like this:
```Python
class ImageClassifier:
    def __init__(self, database):
        pass
        
    def __str__(self):
        pass
        
    @staticmethod
    def createDatabase(imageFolder, databasename):
        pass
        
    def openDatabase(self):
        pass
        
    @staticmethod
    def normalizeBinary(image):
        pass
        
    @staticmethod
    def normalizeNot(image):
        pass
        
    def normalizeDatabase(self, normFunction):
        pass
        
    def classifyImage(self, img, normFunction):
        pass
    
```
Do not worry about the *@staticmethod*-thing there, we will get to that in a bit. 

### Constructor and str-Method
Let's begin with the easy stuff first. An image classifier always works with one database, so our constructor shall define an empty database and the database name. The string-method return a simple description of the class, so with which database the current classifier is working.

```python
    def __init__(self, databaseName):
        self._database = None
        self._databaseName = databaseName

    def __str__(self):
        if not self._database == None:
            return ("ImageRegognizer with database of" + str(len(self._database)) + " different Classes, each containing " + str(len(self._database[0])) + " images.")
```

Okay, that was easy. Let's go on with actually creating such a database.

### Creating the database
We said that a classifier uses a database to classify images. But the creation of such a database is somehow independent of the classifier itself. Creating a database belongs to the ImageClassifier in general, but not to a particular instance of the class. This is why we used the *@staticmethod*-attribute: It defines that we can use this method without having an instance of the class. This is visible by the absence of the *self* argument at the first position: the Method does not know to what instance it belongs to. In fact, it is completely decoupled from ANY instance and just sits in the class itself. But what does that even mean? It expresses that the method underneath does not use any instance-specific fields or methods and can be called freely without an instance. This is super cool because we can then create a new database just by calling
*ImageClassifier.createDatabase(***)*, and then use that database for any particular classifier we want.

Fine, we use a *@staticmethod* then, but what exactly does *.createDatabase()* do now? Well, what do we want to achieve? We want to save a database to the disk containing all images as arrays. Why arrays? Because we will manipulate the images later on, which we can only do when we see them as arrays. So we need to open our images, convert them to arrays and save them as a database. Let's get through that step-by-step. 

#### The Images-schema
The user will provide us with the folder-name in which the images are located. We as developer define how he has to name the images so that we can identify their class. He has to follow the following structure:
>-imageFolder
> ----digit_index.jpg

Meaning in the imageFolder, all images are located and named by the digit they display and some incrementing index. You find an example of an image folder in the github repo, but I also give a quick example here:
>-images
> ---0_0.jpg
> ---0_1.jpg
> ---0_2.jpg
> ...
> ---0_15jpg
> ---1_2.jpg
> ...
> ---9_15.jpg

To initially train your classifier, you will need to create such images by yourself. Pay attention to always draw the digit a little different so that our classifier can classify badly drawn digits as well. 

![Nummber 1](https://github.com/joelbarmettlerUZH/Simplest_Image_Classifier/raw/master/readme_resources/test_images.png)

Fine, can we please code now?? Sure, let's get into it. 

### The actual code
We want to keep it simple here and store the images a dictionary where the keys are the digits and the values are lists containing the array-representation of all according images. 

```pyton
    number_db = {}
    for i in range(10):
        number_db[i] = []
```

Now let's loop through our image folder and open the images by their well-defined name: we assume here that the user always provides 16 images of a kind, so 16 images displaying a zero, 16 displaying a 1 and so forth. We open the image by this naming convention, convert it to an array and insert it into the according list into the dict database:

```python
    for number in range(len(number_db.keys())):
        for index in range(16):
            image = Image.open(imageFolder + "/" + str(number) + "_" + str(index) + ".jpg")
            number_db[number].append(np.array(image).tolist())
```

By calling *np.array(image).tolist()*, we convert our PIL Image object into a python list. The list represents the image, each sublist of the list represents one row, and each sublist of THAT list represents a pixel, while each pixel conists of three values; R(ed), G(reen), B(lue). The following image should make clear how the list conversion looks like.

![Image to list](https://github.com/joelbarmettlerUZH/Simplest_Image_Classifier/raw/master/readme_resources/image_theory_1.png)

Now that we have a dict that serves us as a database, let's save that dict to the disk using pickle. This is a no-brainer. 

```python
with open(databaseName + ".pkl", "wb") as db:
    pickle.dump(number_db,db)
```

In the main function of our script, we can now call this method and create a new database pickle.

```python
if __name__ == "__main__":
    ImageClassifier.createDatabase("images", "number_db")
```

This creates a new *number_db.pkl* in our main directory. 

### Open Database
Okay, now that we have created our database, we need a way to tell our classifier to use it. For that manner, we create an openDatabase method. Why don't we have to specify the database name? Remember that, when we create a new Classifier, we need to specify what database to use, so the name of the database is already an instance variable of the class-instance itself. 

```Python
def openDatabase(self):
    with open(self._databaseName+".pkl", "rb") as db:
        self._database = pickle.load(db)
```
The code is simple: Open the database and load the database content out of the pickle into the self._database variable. 

### Normalize Image
We have talked about normalization a little bit, but not in detail. Why should we need to normalize an image, and what does that even mean? Well, the only thing that we defined about the input image is that is has to have a fixed size of 64 by 64 pixels. But that still leaves open quiet some room for variation: what about the colour of the digit and the background? Can we compare a blue nine on a green background? What about a scanned image, where the image has low contrast and a gray background? This seems hard to handle, but in fact it is easy to reduce these variables. 

Let's have a look how an imput image may look like. 

![Noisy](https://github.com/joelbarmettlerUZH/Simplest_Image_Classifier/raw/master/readme_resources/test8_noisy.jpg)

While it looks similar to the digit 8 from our test images (in fact, it is the same image with some visual styling), it is clear that we can not directly compare this image to our database and hope to get a propper result. It would be much nicer if every image had a similar form.

![Noisy Goal](https://github.com/joelbarmettlerUZH/Simplest_Image_Classifier/raw/master/readme_resources/test8_noisy_goal.jpg)

Let's look at a selection of this image and see how we could achieve this reduction. We take a close shot of the region marked in red (have a close look, it is just one pixel collumn and barely visible)

![Selection](https://github.com/joelbarmettlerUZH/Simplest_Image_Classifier/raw/master/readme_resources/test8_noisy_selcetion.jpg)

This is the same region, but now zoomed in. We can see some of the pixel values, the leftmost yellowish pixel has the RGB value [244, 240, 125], the rightmost pixel has the RGB value [63, 74, 157]. Remember that the yellowish part does not belong to the digit, while the blue part to the right does. 

![collumn](https://github.com/joelbarmettlerUZH/Simplest_Image_Classifier/raw/master/readme_resources/reduction_1.jpg)

Now we would like to modify that pixel array to be able to clearly distinguish what part does belong to the digit, and what part does NOT. As a first step, we get ridd of the colours, because we are actually not at all interested in the these. Why not? Because we can assume that the digit is written on something paper-like, so the digit will have a darker colour than the background, no matter what colour this happens to be. We just want to look at the contrast. So let's convert this image to Greyscale values only by averaging each pixel: [244, 240, 125] becomes 244 + 240 + 125 = 206, therefore [206, 206, 206].

![greyscale](https://github.com/joelbarmettlerUZH/Simplest_Image_Classifier/raw/master/readme_resources/reduction_2.jpg)

Now let's go one step further and make every pixel that is dark fully black, and every pixel that is bright fully white. We make a clear cut: Every pixel that is brighter than 255/2 (255 is the brightest value a pixel can have, 0 the darkest) as bright, every pixel that is darker than 255/2 as dark. We get the following result:

![bw](https://github.com/joelbarmettlerUZH/Simplest_Image_Classifier/raw/master/readme_resources/reduction_3.jpg)

We are almost done! See that pixel values? They are always either fully white [255, 255, 255] or fully black [0, 0, 0]. So we store redundant information. Let's replace [255, 255, 255] with 1 and [0, 0, 0] with 0.

![01](https://github.com/joelbarmettlerUZH/Simplest_Image_Classifier/raw/master/readme_resources/reduction_4.jpg)

We now successfully converted not just a coloured image to greyscale, but a greyscale image to binary. 

![binarydemo](https://github.com/joelbarmettlerUZH/Simplest_Image_Classifier/raw/master/readme_resources/binary_demo.jpg)

### Code please! Now!
Okay, let's realize that in python. First, we loop through every column of the image. Then, we loop through all the pixels in the column. Then, we loop again thorugh every pixel value and sum up their colour values. If the total colour value divided by three happens to be greater than 255/2, we assign the pixel a 1, otherwise a 0. Finally, we return the image. 

```Python
@staticmethod
def normalizeBinary(image):
    for column in image:
        for pixel in range(len(column)):
            total_color = 0
            for color in column[pixel]:
                total_color += color
            if total_color/3 < 255/2:
                column[pixel] = 0
            else:
                column[pixel] = 1
    return image
```

### Normalize the whole Database
Now that we can normalize an image, we can also normalize a database by just looping thorugh it every image it contains and replacing the images with their normalized ones. 

```python

def normalizeDatabase(self, normFunction):
    for number in range(len(self._database.keys())):
        for image in self._database[number]:
    image = normFunction(image)
```

## The classifier itself
Now that so much pre-processing work is done, the classifier itself is pretty straight forward. What we want to do is to open the image, convert it to a numpy array, normalize it with the same normalization function as the database, and then compare it to all the images in the database while keeping track of the similarity between each of them. 
Let's start with opening and converting the images:

```python
def classifyImage(self, img, normFunction):
    test_image = Image.open(img)
    test_image = np.array(test_image).tolist()
    test_image = normFunction(test_image)
```

Now the comparing part starts. First, we initialize a dict in which we will store the similarity between the provided image and each image-class in the database. Then, we loop through the database and compare the two images, pixel for pixel. If the two pixels match, we add two confidence-point to our confidence dictionary for the class which we are comparing to. When the pixels do not match, we substract one confidence point of the compared class. So our confidence dict looks like this:
**confidence_dict = {"0": 55, "1":4, "2":32, "3":120   ...   , "8":432, "9":43}**
It essentially tells us, how many pixels matched in comparison to all the images in that class. Let's assume the input image displayed the digit "8". When comparing it to all the images in the database that display a "0", we may get like 55 confidence points. That seems like a lot, but the absolute value of the number does not matter, because when we compare the input image to the database images displaying the digit "8" as well, we will (hopefully) get a much hihger confidence number, like 432! 

```python
    for i in range(10):
        confidence_dict[i] = 0
    for number in range(len(self._database.keys())):
        for image in self._database[number]:
            for column in range(len(image)):
                for pixel in range(len(image[column])):
                    if image[column][pixel] == test_image[column][pixel] == 1:
                        confidence_dict[number] += 2
                    if image[column][pixel] == test_image[column][pixel] == 0:
                        continue
    confidence_dict[number] -= 1
```

Wait wait wait. Why do we add 2 confidence points for a matching pixel, but only substract 1 for a dismatch? Good question! Well, assume you deal with the digit "1" as an input image, and compare it to another "1" in the database. Let's assume the pictures match nearly perfectly, except one pixel is moved a little bit. See how that generates 2 negative confidence points? So when a pixel does not match, it is treated far worse than when they match. To balance that out we add not just one, but two confidence points for every match. 

![confidence-match](https://github.com/joelbarmettlerUZH/Simplest_Image_Classifier/raw/master/readme_resources/match_demo.jpg)

Now we need a little bit of statistics to get a percentage value back from our confidence array. First, we sort the confidences to find out what our prediction is: the digit with the highest confidence number. But we are not done yet: We also want to indicate how sure we are with our prediction: If we predict the image to be a "1", but we are only 51% sure and it could also be a "7", our prediction is not worth much. To be fair, we should at last show how certain we are. We do that by comparing the certainty of our prediction with the certainty of our second-best guess: When the second guess is nearly as good as our prediction, we are like 50% certain. But if our guess is prediction has a 3 times higher confidency than our second guess, we can be close to 100% sure to really deal with the number we predicted. We devide 1 by our best guess and multiply the outcome with our second best guess, *100 to get a percentage value. Rounded to two digits, we get a nice and clean percentage value back.

```python
    max_confidence = sorted(confidence_dict.values(), reverse=True)
    for key, value in confidence_dict.items():
        if value == max_confidence[0]:
            return key, min(round(((1.0/max_confidence[0])*max_confidence[1])*100,2), 100.00)
```

### Testing our script
Wow, we are finally done! Let's test how good our script performs. Write a main function that is executed when our script is directly called and test the script with some test-images (be sure to not include these in the training images, that would be cheating). I have added some test images named test_*digit*.jpg into the main directory, each displaying a digit between 0 and 9. Let's try it out: Create a new database with our training images, create a new classifier with that database and open the database. Normalize the database with our *normalizeBinary* function, then loop thorugh each of the test images and call *.classifyImage()* with the same normalization Function. 

```python
#Use for testing ImageRecognizer Class
if __name__ == "__main__":
    ImageClassifier.createDatabase("images", "number_db")
    imageClassifier = ImageClassifier("number_db")
    imageClassifier.openDatabase()
    print(imageClassifier)
    imageClassifier.normalizeDatabase(ImageClassifier.normalizeBinary)
    for number in range(10):
        print(imageClassifier.classifyImage("test" + str(number) +".jpg", ImageClassifier.normalizeBinary))
```

The results are pretty good!
![results](https://github.com/joelbarmettlerUZH/Simplest_Image_Classifier/raw/master/readme_resources/results.png)

## But wait!
What if we do not want to normalize the images, but compare the raw-ones? We have to provide a normalization function, it's mandatory in some of the function calls! True, we have to find a way around that. But that's easy, let's just cheat a little bit and create a normalization function that does - nothing. Just return the blank image again. 

```python
    @staticmethod
    def normalizeNot(image):
        return image
```

## Ways to improve
This little script is just the basic of image classifying, right. We could get a lot more out of this algorithm with some tweaks. For example, our algorithm fails right away when it compares a "1" that is in the left half of the image to a "1" that is on the right part, because it can not find any matching pixels. We could solve that by cutting all white space on top and on the left of the digit. Further, our images have to be fixed size, so we could try to make that a bit more flexible by scaling them first. Or we could make the algorithm more efficient by not saving all our images to the database, but sum them up to one, "total-averaged" image of the digit, which would save us a lot of time on bigger training sets. But these improvements are up to you! 

----

# License
MIT License

Copyright (c) 2018 Joel Barmettler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Hire us: [Software Entwickler in Zürich](https://polygon-software.ch)!


