# Mechanistic_Interpretability

Eval number 3:
1. Variable stack length data - VV gud acc -> led us to believe that it was using a stack -> We also probed the model to see if it was using a stack and the results were inconclusive, because the accuracy was varying on the stack depth.
2. More higher value of stack depth data -> lesser acc -> this was not very accurate as the generation of unbalanced bracket strings for high stack depth was not very good and generated very less samples. (standardised length)
3. Added perturbation to generate more unbalanced data with a particular stack depth -> 50% acc (loss doesnt decrease) -> Hypothesis became Transformer DOES NOT USE STACK, but uses counts -> This is because the perturbed data had very low count. The unbalanced strings were very less unbalanced, so the Transformer probably used the count of brackets. The unbalanced strings were very similar to the balanced strings because count bohot kam tha.
4. To test this hypothesis, we generated a dataset with unbalanced strings of count 0 and tested it on the previously trained model -> 0% acc -> Then we trained on the this new dataset as well and saw that the model could still not solve the problem, but it learnt an additional pattern which is that any string starting with a closing bracket is unbalanced. This further supported our hypothesis as it was not able to learn the unbalanced strings with count 0. It is only failing on the case where counting fails and the stack succeeds.

DATA NEEDED:
1.  SCALING DATA VERY LARGE
2. NORMAL TRAINING DATA WITH ALL COUNTS FOR FAIRNESS
TESTING DATA WITH LOW COUNTS -> SHOULD GET LOW ACC
TESTING DATA WITH HIGH COUNTS -> SHOULD GET HIGH ACC

Next Steps:
1. Setting up Mechanistic Interpretability to find where the model is using counts.
2. We will be looking at the attention weights, the embeddings given as outputs of the transformer and the logits given as the final output by the MLP.