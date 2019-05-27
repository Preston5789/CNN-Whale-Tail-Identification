# Machine Learning (2018, spring) Final Project

### Humpback Whale Identification Challenge[1] - Can you identify a whale by the picture of its fluke?

After centuries of intense whaling, recovering whale populations still have a hard time adapting to warming oceans and struggle to compete every day with the industrial fishing industry for food.

To aid whale conservation efforts, scientists use photo surveillance systems to monitor ocean activity. They use the shape of whales’ tails and unique markings found in footage to identify what species of whale they’re analyzing and meticulously log whale pod dynamics and movements. For the past 40 years, most of this work has been done manually by individual scientists, leaving a huge trove of data untapped and underutilized.

In this competition, you’re challenged to build an algorithm to identifying whale species in images. You’ll analyze Happy Whale’s database of over 25,000 images, gathered from research institutions and public contributors. By contributing, you’ll help to open rich fields of understanding for marine mammal population dynamics around the globe.

We'd like to thank Happy Whale for providing this data and problem. Happy Whale is a platform that uses image process algorithms to let anyone to submit their whale photo and have it automatically identified.

## Prerequisites

- tensorflow-gpu 1.6.0

- keras 2.0.8


## Getting Started

Prepare training data, testing data, and pretrained model

    >>>bash data.sh

Run Test

    >>>bash test.sh
    
Train your own model(You can edit train parameters at src/trian.py)

    >>>bash train.sh

## Reference

1. https://www.kaggle.com/c/whale-categorization-playground
