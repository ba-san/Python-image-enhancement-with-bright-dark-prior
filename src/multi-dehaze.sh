#!/usr/bin/bash/

workon WRN

files="/home/daisuke/Workplace/Python-image-enhancement-with-bright-dark-prior/images/*"

for filepath in $files; do
  faname_ext="${filepath##*/}"
  echo $faname_ext
  echo '#########################'

  sed -i -e '160,160d' dehaze.py
  sed -i '160i \    src = '\"$faname_ext\" dehaze.py
  
  python2 dehaze.py

done
