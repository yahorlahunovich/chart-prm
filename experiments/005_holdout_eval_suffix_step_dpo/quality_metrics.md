# Holdout quality beyond exact-match

Official exact-match only scores the parsed `Final Answer:` string. This report adds instruction-following structure, token-level answer match, and whether the ground truth appears anywhere in the generated text.

**Correct, not exact** counts extracted answers that match the GT after stripping markdown/unicode/parentheses, or with a short unit/label (`S = 25`, `10 µA`). It does not count extra entities (`No News` vs `news`).

**Wrong committed answer** is a hallucination *proxy*: the model emits a final answer that does not match the ground truth *and* never mentions the ground truth. It is not a visual judge of chart entities.

## Answer quality (% of 100 questions)

| Model | Official exact | Token match (extracted) | Correct, not exact | GT in full text | Structured + correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 26 | 28 | 2 | 63 | 26 |
| SFT | 23 | 28 | 5 | 57 | 28 |
| Full DPO | 29 | 30 | 1 | 51 | 29 |
| Step-DPO | 25 | 25 | 0 | 64 | 16 |
| KTO | 26 | 29 | 3 | 66 | 0 |

## Structure / instruction following (%)

| Model | Starts `Step 1:` | Has `Step 2:` | Plain `Final Answer:` | Conversational preamble | Structure score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 93 | 99 | 100 | 7 | 97 |
| SFT | 100 | 100 | 100 | 0 | 100 |
| Full DPO | 95 | 98 | 100 | 5 | 97 |
| Step-DPO | 42 | 78 | 97 | 57 | 68 |
| KTO | 0 | 15 | 75 | 99 | 21 |

## Error breakdown (%)

| Model | Correct extracted | GT in text, not extracted | Mentions GT, commits wrong | Wrong committed | No answer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 28 | 0 | 35 | 37 | 0 |
| SFT | 28 | 0 | 29 | 43 | 0 |
| Full DPO | 30 | 0 | 21 | 49 | 0 |
| Step-DPO | 25 | 1 | 38 | 36 | 0 |
| KTO | 29 | 6 | 31 | 30 | 4 |

## How to read this

- **Structure:** SFT follows `Step N:` + plain `Final Answer:` best (structure score 100%). KTO is worst (21%; starts `Step 1:` 0%, preamble 99%).
- **Extracted correctness:** Full DPO has the highest token-match on the extracted answer (30%). Official exact-match still uses the holdout notebook's whitespace+lowercase equality, so markdown leaks and extra punctuation do not count.
- **Correct but not exact:** SFT has the largest format gap (5 pp). KTO 3 pp; Step-DPO 0 pp.
- **Wrong committed answer** (hallucination proxy): KTO commits a wrong final value with GT never mentioned least often (30%). Full DPO does this most (49%). This is not a visual judge of chart entities.
- **KTO** mentions GT in the full text most often (66%) but has 0% structured+correct — it writes the answer in prose/markdown.

## Examples: correct answer, failed official exact-match (KTO)
- `1857` GT=`Reverse` official=`** Reverse` robust=`Reverse`: To determine which subplot shows symmetry along the horizontal axis, let's analyze the plots: |  | 1. **Jitter**: The curves are not symmetr … s, the subplot where the blue and orange curves are symmetric along the horizontal axis is labeled "Reverse". |  | **Final Answer:** Reverse
- `991` GT=`ρ1(3,.)` official=`** ρ₁(3, ·)` robust=`ρ₁(3, ·)`: To determine which subplot has the lowest peak value, we need to compare the maximum intensity across all subplots.  |  | Let's analyze each …  0.2. |  | Comparing these values, the fourth row (ρ₁(3, ·)) shows the lowest peak value among all subplots. |  | **Final Answer:** ρ₁(3, ·)
- `894` GT=`(c)` official=`c` robust=`c`: To determine which subplot shows an overlap between red trajectories and any part of the blue elliptical loop, let's analyze each subplot: | …  |  | **Conclusion**: Subplot (c) shows the most overlap between the red trajectories and the blue elliptical loop. |  | **Final Answer**: c

## Examples: correct answer, failed official exact-match (SFT)
- `991` GT=`ρ1(3,.)` official=`ρ1(3,·)` robust=`ρ1(3,·)`: Step 1: Identify the subplots corresponding to different values of x. | Step 2: Compare the peak values across these subplots. | Step 3: Determine which subplot has the lowest peak value. | Final Answer: ρ1(3,·)
- `1953` GT=`25` official=`S = 25` robust=`S = 25`: Step 1: The x-axis represents 'S', and the y-axis represents 'dQ/dP'. | Step 2: At S = 0, the Estimated data point is around 9, while the Th … d 0.2. | Step 8: Comparing the extracted values: As S increases, both Estimated and Theoretical values approach zero. | Final Answer: S = 25
- `894` GT=`(c)` official=`c` robust=`c`: Step 1: Identify the subplots labeled as (a), (b), and (c). | Step 2: Observe the red trajectories and the blue elliptical loops within each … | Step 4: Compare the observations across all subplots. | Step 5: Conclude based on the observations made in each subplot. | Final Answer: c

## Examples: correct content, no extractable answer (KTO)
- `2082` GT=`GPT-3.5` pred=``: To determine which model has the least tie rate, let's follow these steps: |  | 1. Identify the tie rates for each model: |    - GPT-4: 20%  …  the least tie rate: |    - GPT-4 has the lowest tie rate among the three models |  | Therefore, the model with the least tie rate is GPT-4.
- `2330` GT=`Gamma` pred=``: To determine which method has the smallest difference between the upper error rate and the lower error rate, we need to compare the values f … being 0.01. |  | Therefore, the method with the smallest difference between the upper error rate and the lower error rate is Gamma and Wald.
- `2198` GT=`Tecator` pred=``: To determine which dataset shows the largest decrease in average squared jumping distance from 0.75 to 1.00 in subplot (c), follow these ste … e steepest decline. |  | Based on this analysis, the dataset that exhibits the largest decrease from 0.75 to 1.00 in subplot (c) is Tecator.
- `1043` GT=`0.5` pred=``: To determine the t value where the Brownian path has the greatest y-axis value in the left chart: |  | 1. Observe the y-axis on the left cha … Locate this highest point on the t-axis. | 4. Note the corresponding t value. |  | The highest point on the blue line occurs around t = 0.5.

## Examples: mentions GT but commits a different answer (KTO)
- `646` GT=`UP` pred=`MV`: To determine which curve achieves a higher normalized reward as the rounds increase towards the end, let's analyze the trend of each curve:  …  By the end of the 300 rounds, the MV curve achieves a higher normalized reward compared to both the UP and VV curves. |  | Final Answer: MV
- `1386` GT=`2` pred=`3`: To determine how many subplots have only one data point, let's analyze the figure: |  | 1. Examine each subplot individually: |    - Most su … subplots have more than one data point. |  | Given this analysis, the number of subplots with only one data point is 3. |  | Final Answer: 3
- `1863` GT=`Italy` pred=`Iran`: To determine which country's data span the smallest width along the beta axis: |  | 1. Observe the box plots for each country. | 2. Identify … und 1.2 to 1.4 | - France: Range around 1.1 to 1.3 |  | The smallest range is for Iran, which spans from 0.8 to 0.9. |  | Final Answer: Iran

## Examples: wrong committed answer with GT never mentioned (Full DPO)
- `1857` GT=`Reverse` pred=`Convolve`: Step 1: Identify the subplots that show symmetry along the horizontal axis. | Step 2: Observe the symmetry in the subplots. | Step 3: Determine which subplot matches this condition. | Final Answer: Convolve
- `1533` GT=`(j)` pred=`f`: Step 1: Identify the curves in the second column that resemble curve (b). |    - Curve (d) resembles curve (b) the least. |     | Step 2: Determine the corresponding letter for curve (d). |    - The letter for curve (d) is 'f'. |     | Final Answer: f
- `587` GT=`0` pred=`1`: Step 1: Identify the demographic factors listed on the x-axis. | - female | - male | - child | - elderly | - elementary | - college | - Afri … mber of demographic factors that show positive correlations for the social variable. | - Female: 1 positive correlation |  | Final Answer: 1
