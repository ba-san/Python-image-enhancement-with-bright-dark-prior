# Python-image-enhancement-with-bright-dark-prior

This repository is a python implementation of [Nighttime low illumination image enhancement with single image using bright/dark channel prior](https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-018-0251-4).  
Some of the codes are different from the original paper because some descriptions there are missing & wrong.  

## How to Run  
You need to modify inside `if __name__ == '__main__':` in _dehaze.py_ for image, directory and parameters setting.   
You can use _multi-dehaze.sh_ for multi-image processing.  

## Examples 
|Input|Output|
|---|---|
|<img src="https://user-images.githubusercontent.com/44015510/78751084-84f1db00-79ac-11ea-8e09-cbe382bc50b1.png" width="300">|<img src="https://user-images.githubusercontent.com/44015510/78751178-b1a5f280-79ac-11ea-8456-05295841102d.png" width="300">|   
|<img src="https://user-images.githubusercontent.com/44015510/78751294-e5811800-79ac-11ea-91f9-783418b08c75.png" width="300">|<img src="https://user-images.githubusercontent.com/44015510/78751315-f2057080-79ac-11ea-8279-90fe4cc7bc4b.png" width="300">|

## Limitation
I noticed the result is quite bad if spotlights in image.  
|Input|Output|
|---|---|
|<img src="https://user-images.githubusercontent.com/44015510/78845943-07c67480-7a45-11ea-8e13-1a8a34911b1e.jpeg" width="300">|<img src="https://user-images.githubusercontent.com/44015510/78845957-16ad2700-7a45-11ea-8704-01e1e38e2230.png" width="300">|   
|<img src="https://user-images.githubusercontent.com/44015510/78845968-1b71db00-7a45-11ea-937a-29396dd74d94.jpeg" width="300">|<img src="https://user-images.githubusercontent.com/44015510/78845971-1f056200-7a45-11ea-973c-e65afd9e261d.png" width="300">|

## Acknowledge
The codes are based on the following repository.  
https://github.com/joyeecheung/dark-channel-prior-dehazing
