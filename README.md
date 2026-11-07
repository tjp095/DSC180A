### Project Checkpoint Code

The repo contains two different folders, practice and rotated-MNIST. The practice folder contains the assignments that were completed using the neuron data from the Allen Institute, where neuron unit firings were recorded using an electrode insertion into mice across different types of visual stimuli. 

The rotated-MNIST contains the data, setup, and starter code for the rotated MNIST dataset that will be used for the quarter 1 project. 

## Data Access and running notebooks

1. In order to access the data, start by creating and activating the environment using the following lines. 

conda env create -f environment.yml -n allensdk-test
conda activate allensdk-test

2. Download the recordings

python ecephys_setup.py
If this doesn't work, try conda run --no-capture-output -n allensdk-test python ecephys_setup.py

Wait a couple minutes until the session data starts downloading and finishes for both sessions. The terminal should print: 
Session 715093703 ready (884 units)
Session 750749662 ready (761 units)

3. Now the notebooks in the practice folder are ready to run! (The data access ipynb can be skipped)

## AI use

### Code implementations/bug-fixing
AI was mainly used in the implementation-level details and bug-fixes, with high-level structure and function design all done by without resorting to using AI.

*Example prompt:* I have this helper function that should take the rate_df and sort the units by orientation. sorted_unit_ids gives the order of the units I want them in, but I don't know how to make the rows of rate_df conform to it.
```
def sort_by_orientation(spike_df, rate_df):
    sorted_unit_ids = (spike_df[["orientation", "unit_id"]]
        .drop_duplicates()
        .sort_values(by=["orientation", "unit_id"])["unit_id"]
    )
```

*Example response*: Short answer — reindex the columns (units) with sorted_unit_ids:
```
def sort_by_orientation(spike_df, rate_df):
    sorted_unit_ids = (
        spike_df[["orientation", "unit_id"]]
        .drop_duplicates()
        .sort_values(by=["orientation", "unit_id"])["unit_id"]
        .tolist()
    )

    # rate_df: rows = trials, cols = units → reorder columns
    return rate_df.reindex(columns=sorted_unit_ids)
```
**Why `.reindex(columns=...)` is the right tool**

It reorders existing columns to match sorted_unit_ids.

- Any unit in sorted_unit_ids but missing from rate_df becomes a NaN column.

- Any unit in rate_df but not in sorted_unit_ids is dropped.

If you'd rather keep only units present in both (no NaN columns), use `.loc`:
```
common = [u for u in sorted_unit_ids if u in rate_df.columns]
return rate_df[common]
```

### Honing our understanding
AI was also used for sharpening our understandings of the various tools we were given, especially when given dense papers to read every week. 

*Example prompt 1*: I am in a neuroscience / data science course right now, and we are studying brain data from mice. We are going over representational geometry/RSA/CKA techniques for quantifying similarity between systems, in this case different regions of the mouse visual cortex. I would like some help digesting what this week's assignment is asking us to do.

*Example response 1*: I'd be happy to help you digest the assignment! However, I don't see the actual assignment text in your message — you described the course context (mouse visual cortex, RSA/CKA, representational geometry) but didn't paste the specific questions or instructions.

Could you share:

- The assignment prompt itself (paste the text, or describe it in detail)

- What data you're working with — e.g., is it calcium imaging, electrophysiology, fMRI-like? Which regions (V1, LM, AL, PM, etc.)? What's the format (neurons × stimuli × trials)?

- What tools/languages you're expected to use (Python/NumPy, MATLAB, scikit-learn, custom RSA toolbox, etc.)

- Any specific points of confusion — is it the conceptual framing (what RSA/CKA mean), the math, the coding, or interpreting results?

*Example prompt 2*: One area I am hazy on is from the results of this paper: *link to original CKA paper*. From my understanding, CKA asks the question: is the way that model A represents the given stimuli analogous to the way model B represents the same stimuli?

*Example response 2*: Yes — that's exactly the right framing, and it's worth sharpening precisely because "analogous" is doing a lot of work in that sentence. The precise version of your statement: CKA asks do the pairwise similarity structures among stimuli agree between the two systems, after allowing for rotations and uniform scaling of each system's neural space?

**Two things to unpack:**

"Pairwise similarity structures among stimuli." For each system, you can form a K×K matrix of similarities between stimuli (like the cosine matrix from Step 1). CKA is comparing the geometry encoded in those matrices — not the raw neural responses, and not the neuron-to-neuron correspondence.

"After allowing for rotations and uniform scaling." This is the invariance that makes CKA useful for cross-region comparison. If V1 represents orientation in a way that's a rotated, rescaled version of how the HOA represents it, CKA = 1. You don't need neuron 17 in V1 to "mean" the same thing as neuron 17 in the HOA — you just need the shape of the representational space to match.
What CKA does not ask

- It doesn't ask whether individual neurons correspond across regions.

- It doesn't ask whether the two regions have the same number of neurons, or the same firing rates.

- It doesn't ask whether the representations are identical — only whether they're equivalent up to orthogonal transformation and isotropic scaling.

- It's not invariant to arbitrary linear transforms — a shear or non-uniform stretch of the neural space will change CKA. That's a feature: it means CKA is sensitive to distortions of the geometry that aren't just rotations.

**The analogy that might help**

Think of each region's representation as a constellation of K points in N-dimensional space, where each point is one orientation's population vector.

- Two constellations could be related by a rotation (same shape, different orientation in space) → CKA = 1.

- Two constellations could be related by a uniform scaling (same shape, different size) → CKA = 1.

- Two constellations could have the same shape but different relative distances between points (e.g., one stretches along one axis) → CKA < 1.

- Two constellations could be completely unrelated → CKA near chance (which, as we discussed, is not 0).

So CKA is essentially asking: are these two point-clouds the same shape?