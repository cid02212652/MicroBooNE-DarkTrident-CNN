#!/bin/bash

alias e="emacs -nw"
alias l="ls -lrth"

export LC_ALL=C

# Set up ROOT
echo "Setting up ROOT..."
cd /opt/root
source bin/thisroot.sh
cd -

# Set up larcv2 
echo "Setting up larcv2..."
export LARCV_BASEDIR=/usr/dependencies/larcv2/build
export LARCV_LIBDIR=$LARCV_BASEDIR/lib
export LD_LIBRARY_PATH=$LARCV_LIBDIR:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/dependencies/larcv2/python:$PYTHONPATH
export PYTHONPATH=$LARCV_LIBDIR:$PYTHONPATH
source /usr/dependencies/larcv2/configure.sh

# This directory points to where the DM-CNN repo is 
echo "Setting up DM-CNN..."
source setup.sh

alias ipn='ipython notebook --no-browser'
alias l='ls -lrth'
alias e='emacs -nw'

#Color setup
export TERM=xterm-256color
RS="\[\033[0m\]"    # reset
HC="\[\033[1m\]"    # hicolor
UL="\[\033[4m\]"    # underline
INV="\[\033[7m\]"   # inverse background and foreground
FBLK="\[\033[30m\]" # foreground black
FRED="\[\033[31m\]" # foreground red
FGRN="\[\033[32m\]" # foreground green
FYEL="\[\033[33m\]" # foreground yellow                                          
FBLE="\[\033[34m\]" # foreground blue                                            
FMAG="\[\033[35m\]" # foreground magenta                                         
FCYN="\[\033[36m\]" # foreground cyan                                            
FWHT="\[\033[37m\]" # foreground white                                           
BBLK="\[\033[40m\]" # background black                                           
BRED="\[\033[41m\]" # background red                                             
BGRN="\[\033[42m\]" # background green                                           
BYEL="\[\033[43m\]" # background yellow                                          
BBLE="\[\033[44m\]" # background blue                                            
BMAG="\[\033[45m\]" # background magenta                                         
BCYN="\[\033[46m\]" # background cyan                                            
BWHT="\[\033[47m\]" # background white                                           

PS1="$HC$FRED$FCYN${debian_chroot:+($debian_chroot)}\u$FWHT: $FYEL\W$FRED :\\$ $RS"
#PS1='\w\$ '
PS2="$HC$FRED&gt; $RS"

ulimit -n 2048
