
echo "Enter a directory"
read dir
#echo "This email is $(python Spam4.py $dir)" #for single email
python SpamBash.py $dir #for folder of emails
