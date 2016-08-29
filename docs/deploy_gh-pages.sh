#!/bin/bash

DOCSDIR=$(pwd)
TMPDIR=~/tmp_docs
DSTDIR=$(cd $DOCSDIR/.. && pwd)

echo
echo "Preparing documentations for deployment to gh-pages..."
echo "======================================================"
echo

echo "Removing $TMPDIR to make sure it is recreated fresh & clean."
rm -rf $TMPDIR

echo
echo "Removing old data to prevent any rebuilding artifacts."
rm -rf build

echo "Remaking the docs..."
make html

echo
echo "Coping new docs to $TMPDIR and switching branch to gh-pages"
cp -pRv build $TMPDIR

echo
git checkout gh-pages

echo
echo "Removing old data before moving html files of the new docs in"
echo "from $TMPDIR/html"
cd $DSTDIR
rm -rfv .nojekyll .buildinfo _* *.html *.inv packages *.js  # docs pySW4

# cp -pRv $TMPDIR/html/{.[!.],}* $DSTDIR
cp -pRv $DOCSDIR/build/html/{.[!.],}* $DSTDIR
rm -rfv docs pySW4

echo
echo "Adding new doc files to be commited, commiting changes"
git add -A
git commit -m "Generated gh-pages for `git log master -1 --pretty=short --abbrev-commit`" && git push origin gh-pages

echo
echo "Switching back to the master branch..."
git checkout master

echo
echo "Putting the docs back in $DOCSDIR/build..."
mkdir $DOCSDIR/build
cp -pRv $TMPDIR/* $DOCSDIR/build

echo
echo "Cleaning up..."
rm -rfv $TMPDIR

echo
echo "Done!"
