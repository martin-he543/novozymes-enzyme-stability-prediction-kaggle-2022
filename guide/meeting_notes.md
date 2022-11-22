# Meeting Notes

## 22.11.2022 Meeting 1

**Rubén:**
1. Use Transfer learning with AlphaFold.
2. Copying the first few layers of AlphaFold with their parameters as they are.
3. Discard the rest of the network.
4. Give the network new randomised layers that we will retrain (as output shape is different).
5. For first couple cycles of training, original layers are frozen.
6. Unfreeze the layers for the last few epochs.

