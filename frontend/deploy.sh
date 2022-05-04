#!/usr/bin/env sh

# abort on errors
set -e

# build
quasar build

# navigate into the build output directory
cd dist/spa

git init
git add -A
git commit -m 'deploy'

git push -f git@github.com:lashhw/AI-Final-Project.git main:gh-pages

cd -