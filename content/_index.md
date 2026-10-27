---
title: Home
---
### About
ML Researcher with an academic background in Statistics and Mathematics. Currently researching evaluation-awareness and kinds of metagaming in language models, and how Inoculation prompting shapes learning.

Other things I'm thinking about:
- Unverbalized evaluation awareness in language models: While VEA is easy to catch, I'm interested in the science behind techniques that reliably catch unverbalized eval-awareness as IMO, it's one of the biggest challenges faced by the science of alignment right now.
In my current research, I am looking into the interplay between models' eval belief P(eval), rewriting evals to make them more realistic, how that changes behaviour and affects VEA. So far, it's been hard to synthetically produce evals that significantly change P(eval) without which studying metagaming correctly isn't possible.
    - More broadly, this direction deals with CoT faithfulness which also has implications for monitoring. And my current understanding is the CoT traces are going to be less and less helpful for any kind of monitoring as capabilities increase eg. a spike in Astra's no CoT capabilities, CoT obfuscation (situational awareness can make this worse).

- How language models generalize, specifically during RL-based posttraining and its implications for models' safety.

- Models are becoming increasingly situationally aware. It's a double edged sword - this can be a useful property for alignment researchers to leverage but can also lead to unexpected and unintended generalization heavily increasing the risk of many failure modes like sabotage, scheming etc.
    - What's the solution? Make models less situationally aware? Build better, more realistic evals to catch consequences before deployment? Develop training techniques that teach models to behave alike regardless of the current situation (eval vs deployment)? Train generally good models + design better specification strategies making sure models comply with developers' intended training signal?

- Other than my technical safety research interests, I love to [learn](https://srishti-git1110.github.io/blog/matmul-cpu/) about performance engineering and how computers work. 


I have in mind a lot of open ideas and problems + concrete plans to tractably study many of them but am bottlenecked by funding. If you're looking to collaborate, have funding for a technical safety problem you want to see progress on, I'm very interested in listening to ideas of all kinds. 👇


Connect: [twitter](https://twitter.com/srishti_gureja) or [email](mailto:srishtigureja1110@gmail.com).
