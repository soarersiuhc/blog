#!/bin/bash

# --title: title
# --date: date
# --draft: draft

# variables assingment/argument parsing
date=$(date +%F)
draft=false

for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$KEY" in
            --title)            title=${VALUE} ;;
            --date)             date=${VALUE} ;;     
            --draft)            draft=true ;;     
            *)   
    esac    
done

escaped_title=${title//[ :]/-}

# script begins
if $draft; 
then 
	loc="./_drafts/${date}-${escaped_title}.markdown"
else
	loc="./_posts/${date}-${escaped_title}.markdown"
fi
 
echo "Copying template to ${loc}"
cp _ignore/template.markdown $loc
sed -i "" -e "
s/{{title}}/$title/g
s/{{date}}/$date/g
" $loc
