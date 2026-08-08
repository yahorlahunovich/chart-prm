# Judge Fail Analysis — Error Categories
- Fail analyses scored: **2920**
- Method: priority regex taxonomy on judge `analysis` text; KMeans(k=8) on MiniLM embeddings as secondary check.

## Primary category distribution

| Category | Count | Share |
|---|---:|---:|
| Incomplete / truncated reasoning | 33 | 1.1% |
| Hallucinated entity / label not on chart | 445 | 15.2% |
| Arithmetic / calculation mistake | 38 | 1.3% |
| Axis / layout / chart-structure misread | 702 | 24.0% |
| Wrong ranking / extremum (highest/lowest/second) | 319 | 10.9% |
| Wrong series / color / legend identity | 568 | 19.5% |
| Bad comparison / threshold logic | 179 | 6.1% |
| Wrong numeric value read from chart | 181 | 6.2% |
| Logic inconsistency / false conclusion | 240 | 8.2% |
| Other / unspecified | 215 | 7.4% |

## Category exemplars (what the judge says)

### Incomplete / truncated reasoning (n=33)
- The step is truncated to 'Summaring Step 2 and' with no completed reasoning or conclusion, providing no verifiable claim and breaking logical coherence.
- The step misreads values on the rightmost plot (e.g., T1 at β=1 is ~38-45%, not ~55%) and contains garbled incomplete entries, failing to report T4 at β=0 and T2/T3 deltas. It also incorrectly concludes 'highest Accuracy Value is T4' (absolute value, false as T2 is ~70% at β=0) instead of computing increase Δ=acc(1)-acc(0).
- Duplicate inaccurate description of values and truncates without performing required ShapeWorld vs Birds Concept variance comparison, hallucinating content instead of answering question.
- Step 0 claims yellow squares exist only for brushingTeethElectric, dishes, nothing, and typing, missing at least 9 other yellow cells visible in the chart such as blender (1), brushingTeethRegular (3,1,1), medicine (2), microwave (4), shaving (1), and typing's second yellow. This is an incomplete and incorrect identification of yellow squares.

### Hallucinated entity / label not on chart (n=445)
- Step 0 misreads the y-axis as 'V' while the chart shows ~n-bar, and hallucinates a 'red line' – the split boundaries are gray dashed lines. It also conflates steps with redundant phrasing.
- Step 1 repeats the 'red line' hallucination which does not exist in any subplot, and provides no valid extraction of split direction (horizontal in a,b,d vs vertical in c). Thus factually incorrect.
- At Degree 100, the orange (Non-top institutions) bar is near zero (around 0.01 or less), not 0.23. The claimed value of 0.23 is a hallucination and misreads the y-axis.
- Repeats the hallucinated value 8,000 as maximum return, which is not present on chart and conflicts with verified answer 750.

### Arithmetic / calculation mistake (n=38)
- Step 3 contains incorrect arithmetic and conclusion: even using its own 98.47, 98.47-89.25=9.22 not 6.47, and using chart truth 98.42-89.25=9.17, not 6.47%, so logical consistency fails.
- Claims 200 = 300 which is mathematically false, and text is incoherent. Comparison does not correctly rank medians and fails to identify second lowest.
- While 60% > 40% is true arithmetic, concluding T3 has the largest increase is false and contradicts the chart where T4 rises from ~30% to ~70% and ground truth is T4.
- The inequality chain 0.998 > 0.997 > 0.999 is mathematically false, since 0.997 is not greater than 0.999, making the comparison logic invalid.

### Axis / layout / chart-structure misread (n=702)
- Misreads the value and unit: claims around 800 steps at 10h, but y-axis is Episode return not steps, and ground truth/chart show ~750. The 800 estimate is inaccurate.
- The x-axis is correctly identified as 'Time (s)', but the step incorrectly reframes the task as finding the 'second-highest accuracy' instead of the second-highest time, which misinterprets the question.
- Factually incorrect reading of the chart: the horizontal dashed lines in (a), (b), (d) are at ~10^-3 to 10^-2, not at 10^-5/10^-4, and mischaracterizes the split values for green/blue points. Subplot (c) shows a vertical, not horizontal, boundary.
- Incorrect conclusion contradicts the chart and ground truth. (a), (b), and (d) all show horizontal top/bottom splits, while (c) shows a vertical left/right split, so (c) is the outlier, not (d).

### Wrong ranking / extremum (highest/lowest/second) (n=319)
- Fails: misidentifies second-highest time as 5 seconds and incorrectly places MobileNetV2 (α=0.5) at 5s with 91%; chart shows that model at <1s and second-highest is Xception at ~5.7s.
- Fails: claims 6 seconds is the second-highest time, but 6s is the highest time (DenseNet201). Accuracy of ~92% for DenseNet201 is approximately correct, but ranking is wrong.
- DenseNet201 is at 6s (the highest time) with 92% accuracy, not slightly lower than 6s. The step misreads the time position, confusing DenseNet201 with Xception.
- Xception is the second-highest time (~5.8s), but its accuracy is 94% on the chart, not 93% as stated, which directly contradicts visual data and ground truth 94.

### Wrong series / color / legend identity (n=568)
- The histogram clearly shows blue (Top at 50, ~0.05-0.07) higher than orange (Non-top at 100, ~0.01), so claiming 'no clear indication' is factually false and contradicts the ground truth that Top institutions is higher.
- At t=10h in hopper:stand the distributed (darker) curves are equal or higher than single-process and converge near ~750, so claiming single-process is higher misreads the legend/colors.
- Visually at Degree 50 the blue bar (top institutions) is around 0.04-0.06, not slightly above 0.10, so this misreads the chart value.
- The chart shows only CS and PHY have orange micro medians above 0.7, so claiming three categories is factually incorrect and contradicts the ground truth answer of 2.

### Bad comparison / threshold logic (n=179)
- The step introduces undefined 'Category A/B' instead of Top/non-top institutions and incorrectly claims distributions are 'below 50' and 'above 100', contradicting the histogram where both extend beyond those points with bulk near 0.
- Step incorrectly switches to 'Macro category' and claims only one category is above 0.7, contradicting Step 1 and the chart where micro medians for CS and PHY are above 0.7, making it factually and logically wrong.
- Step 1 misreads the chart: CS (~0.8) and PHY (~0.85) have micro medians above 0.7, but CMP (~0.4) and PHO (~0.56) are well below 0.7. Claiming three categories CS, CMP, PHO is factually incorrect.
- Misreads the chart: CS micro median ~0.8 and PHY micro median ~0.9 are above 0.7, but CMP micro median is ~0.4-0.5, not above 0.7. Claiming three categories including CMP is factually incorrect.

### Wrong numeric value read from chart (n=181)
- The step incorrectly includes multiple methods whose medians are >0.7 (e.g., UNIFORM (CE, MSE) ~0.77, SOFTADAPT (CEPred, MSE) ~0.78, LEARNABLE (CE, MSE) ~0.81) while claiming median <0.7, misreading the chart.
- At 4000, Cosine + CLP(LASER) is ~40%, not ~42%; the claimed value shows no increase from the previous (incorrect) 42%.
- At 4000, Cosine + LASER dashed line is ~41% constant, not ~39% as claimed.
- Step 1 misreads the chart: Class pair 3 and 8 at no noise is 98.42 per labels (not 98.47), and Class pair 1 and 9 at no noise is ~98.13, not 94.28% which is closer to its value at ε=1.22.

### Logic inconsistency / false conclusion (n=240)
- ROUND-ROBIN-COMBO shows the narrowest box/whisker spread among methods with median <0.7, not Uniform (MSE), so the least variability conclusion is factually wrong.
- Conclusion repeats the erroneous 6800 value, which contradicts the verified ground truth of ~750, so it is factually incorrect despite being logically consistent with Step 2.
- Point (0,-1) has distance 1 from (0,0), equal to the radius, so it lies exactly ON the circle, not outside or within the radius. The statement is false and self-contradictory.
- Concludes no steepest increase based on the false ~30 values; contradicts chart where Cosine+CLP(LASER) clearly increases from ~35 to ~40 while others are flat, and contradicts ground truth answer.

### Other / unspecified (n=215)
- Claims the point is slightly beyond the radius and neither inside nor outside, which is false; (0,-1) is exactly on the circle with radius 1.
- Step compares absolute values (42% > 41% > 39%) instead of computing steepest increase (value4000 - value2000); ground truth is Cosine + CLP with ~5% rise from ~35% to ~40%.
- Repeats false equality 200 = 300 and garbled conclusion. Does not correctly compare medians nor identify port 53, contains mathematical error.
- In the rightmost plot T4 actually increases sharply from ~35% to ~60%, while T2 decreases from ~70% to ~50%. Claiming T2 increases to 88% and T4 decreases to 40% is opposite to the true trends.

## Embedding KMeans discovery view
Useful when a rule category is broad; clusters show recurring phrasings.

### KMeans cluster 0 (n=377, majority rule label: Wrong series / color / legend identity)
- Top terms: blue, green, red, legend, color, yellow, orange, line
- While yellow line values are indeed <0.9, the reference to τ=1 is hallucinated (axis is 200-1000) and it inherits the false premise that blue is highest, contradicting the chart where red is highest.
- Asserts Green's closest match is 0.3, which is factually false; the rightmost plot shows red peaks near 0.4-0.5 and blue peaks near 0.3, not green, so the final conclusion is wrong.

### KMeans cluster 1 (n=428, majority rule label: Wrong ranking / extremum (highest/lowest/second))
- Top terms: ground truth, ground, truth, highest, lowest, chart, contradicts, factually
- Comparison 0.01 > 0 is mathematically true but factually inaccurate per chart because it perpetuates the incorrect differences and implies Jul 21 is largest, contradicting ground truth Nov 20-Jan 21 where true visual gap is maximal.
- Concludes the green line at 550m is the lowest, which is factually false per the chart and ground truth (600m is the lowest). The conclusion inherits the prior misread and misidentifies the lowest point.

### KMeans cluster 2 (n=230, majority rule label: Axis / layout / chart-structure misread)
- Top terms: subplot, subplots, ground, ground truth, truth, factually, line, step
- Factually incorrect per the chart – in the left subplot the blue solid line starts on top but is crossed and overtaken by the yellow dashed line, so it is not consistently above others, which matches ground truth that no line is consistently above.
- FAIL – In the first subplot the highest peak is the green line (Actual, ground truth) reaching ~0.0015, not the orange line, and the quoted value ~0.0002 misreads the y-axis scale which goes up to 0.0015.

### KMeans cluster 3 (n=386, majority rule label: Wrong series / color / legend identity)
- Top terms: curve, line, blue, curves, 10, intersection, near, point
- Step 1 is factually incorrect; the chart shows the red curve peaks at x=1.0 (coincident with the green vertical line), not at 1.9, contradicting the visual data and ground truth of 1.
- Step 1 misreads the visual data: the red curve peaks exactly at the green vertical line where x=1.0, not at approximately 1.3, contradicting the chart and the ground truth value of 1.

### KMeans cluster 4 (n=235, majority rule label: Wrong series / color / legend identity)
- Top terms: year, shows, chart, 2010, peak, 2020, incorrect, chart shows
- This conclusion is false because the chart also shows years 2008 (7) and 2018 (7) exceeding 6, which were not accounted for.
- The chart starts at 2020-07 so June 2020 is not even displayed, and the most significant volatile swings in both dotted lines occur around 2021-03 up to 0.16, not mid-2020, contradicting both visual data and ground truth.

### KMeans cluster 5 (n=333, majority rule label: Logic inconsistency / false conclusion)
- Top terms: step, truth, ground truth, ground, contradicts, answer, concludes, incorrect
- Final count of 3 matches ground truth, but the justification includes the false -6 point and omits the third true intersection around 5-7s in -10 to 10 range, so the step contains incorrect data.
- Concludes 6 steps, which is incorrect; the ground truth is 4 steps. It propagates the counting error from the previous step.

### KMeans cluster 6 (n=499, majority rule label: Wrong series / color / legend identity)
- Top terms: chart, value, near, values, 10, shows, ground truth, ground
- In the top-left Double DID chart at minimum time (~-2), the point estimate is around -0.02 and the shaded lower bound is about -0.05 to -0.06, not -0.1, so -0.1 is a substantial misread.
- Misreads the peak location as t3≈30; the chart shows the tall central peak at t3≈40 per ground truth, so the value is factually incorrect.

### KMeans cluster 7 (n=432, majority rule label: Axis / layout / chart-structure misread)
- Top terms: axis, chart, axes, step, labeled, factually, label, labels
- The step claims the x-axis represents levels s_0, s_1, s_2, but the chart shows levels on the vertical/y-axis with horizontal lines labeled s_0, s_2, s_1, t, 0. This axis misidentification is factually incorrect.
- Step reverses the axes. The chart shows issues listed vertically on the y-axis and Count on the x-axis, not the other way around, making the description factually incorrect.

## Takeaways
- Dominant failure modes: Incomplete / truncated reasoning (1%); Hallucinated entity / label not on chart (15%); Arithmetic / calculation mistake (1%).
- Early steps are enriched for axis/layout and series/color identity errors; later steps pick up more comparison/ranking/conclusion failures.
- Arithmetic is rare in judge text (~1%); most errors are visual grounding (what was read) rather than pure math.
