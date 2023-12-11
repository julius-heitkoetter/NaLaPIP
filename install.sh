#!/bin/bash
#---------------------------------------------------------------------------------------------------
# Install the NaLaPIP Environment
#---------------------------------------------------------------------------------------------------
# Configure and install

# generate the setup file
rm -f setup.sh
touch setup.sh

# Where are we?
HERE=`pwd`

# This is the full setup.sh script
echo "# DO NOT EDIT !! THIS FILE IS GENERATED AT INSTALL (install.sh) !!
export BASE=$HERE
export OPENAI_API_KEY=\`cat $HOME/.openai/api.key\`
export PYTHONPATH=\${PYTHONPATH}:\${BASE}
" > ./setup.sh