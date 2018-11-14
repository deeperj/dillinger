#+Title: Deep Neural Network Techniques in Low Resource Speech Recognition


# Introduction 
Automatic Speech Recognition is a subset of Machine Translation that takes a sequence of raw audio information and translates or matches it against the most likely sequence of text as would be interpreted by a human language expert.  In this thesis, Automatic Speech Recognition will also be referred to as ASR or speech recognition for short.

It can be argued that while ASR has achieved excellent performance in specific applications, much is left to be desired for general purpose speech recognition. While commercial applications like google voice search and Apple Siri gives evidence that this gap is closing, there is still yet other areas within this research space that speech recognition task is very much an unsolved problem.

It is estimated that there are close to 7000 human languages in the world and yet for only a fraction of this number have there been efforts made towards ASR.  The recognition rates that have been so far achieved are based on large quantities of speech data and other linguistic resources used to train models for ASR. These models which depend largely on pattern recognition techniques degrade tremendously  when applied to different languages other than the languages that they were trained or designed for.  In addition due to the fact that collection of sufficient amounts of linguistic resources required to create accurate models for ASR are particularly laborious and time consuming to collect often extending to decades, it is therefore wise to consider alternative approaches towards developing systems for Automatic Speech Recognition in languages lacking the resources required to build ASR systems using existing mechanisms.

## ASR as a machine learning problem \label{ASRMLP}
Automatic speech recognition can be put into a class of machine learning problems described as sequence pattern recognition because ASR attempts to discriminate a pattern from the seqeuence of speech utterances. 

One immediate problem realised with this definition leads us to a statistical speech models that describes how to handle this problem described in the following paragraph.

Speech is a complex phenomena that begins as a cognitive process and ends up as a physical process.  The process of automatic speech recognition attempts to retrace the steps back from the physical process to the cognitive process giving rise to latent variables or mismatched data or loss of information from interpreting speech from one physiological layer to the next.

It has been acknowledged in the research community \citep{2015watanabe,deng2013machine} that work being done in Machine Learning has enhanced the research of automatic speech recognition.  Similarly any progress made in ASR usually constitutes a contribution to enhances made in machine learning algorithm.  This also is an attribution to the fact that speech recogntion is a sequence pattern recogntion problem.  Therefore techniques within speech recognition could be applied generally to sequence pattern recognition problems.

The two main approaches to machine learning problems historically involve two methods rooted in statistical science.  These approaches are the generative and discriminitative models.  From a computing science perspective, the generative approach is a bruteforce approach while the discriminative model is the more heuristic method to machine learning. This chapter establishes the basic definitions of these two approaches in order to introduce the reasoning behind the methods used in this research for low resource speech recognition as well as introduces the Wakirike language speech recogniser as the motivating language under study.

### Generative versus Discriminative speech model disambiguation
In the next chapter, the Hidden Markov Model (HMM), is examined as a powerful and major driver behind generative modeling of sequential data like speech.  Generative models are data-sensitive models because they are derived from the data by accumulating as many different features which can be seen and make generalisations based on the features seen. The discriminative model, on the other hand, has a heuristic approach to make classification.  Rather than using features of the data directly, the discriminative method attempts to characterise the data into features. It is possible to conclude that the generative approach uses a bottom-to-top strategy starting with the fundamental structures to determine the overall structure, while the discriminative method uses a top-to-bottom approach starting with the big picture and then drilling down to discover the fundamental structures.

Ultimately the generative models for machine learning learning can be interpreted mathematically as a joint distribution that produces the highest likelihood of outputs and inputs based on a predefined decision function.  The outputs for speech recognition being the sequence of words and the inputs for speech being the audio wave form or equivalent speech sequence.

$$d_y(\mathbf{x};\lambda)=p(\mathbf{x},y;\lambda)=p(\mathbf{x}|y;\lambda)p(y;\lambda)$$
\label{eqn1_1}

where $$d_y(\mathbf{x};\lambda)$$ is the decision function of $$y$$ for data labels $$\mathbf{x}$$.  This joint probability expression given as $$p(\mathbf{x}|y;\lambda)$$ can also be expressed as the conditional probability product in equation_ref.  In this above equation, lambda predefines the nature of the distribution \cite{deng2013machine} referred to as model parameters.  

Similarly, machine learning discriminative models are described mathematically as the conditional probability defined by the generic decision function below:
$$d_y(\mathbf{x};\lambda)=p(y|\mathbf{x};\lambda)$$
\label{eqn1_2}

It is clearly seen that the discriminative paradigm is a much simpler and straight forward paradigm and indeed is the chosen paradigm for this study.  However, what the discriminative model gains in discriminative simplicity it loses in model parameter estimation($$\lambda$$) complexity in equation (\ref{1_1}) and (\ref{1_2}).  As this research investigates, the although the generative process is able to generate arbitrary outputs, its major draw back is the direct relation to the training data from which the model parameters are learned. Specific characteristics of various machine learning models are reserved for later chapters albeit the heuristic nature of the discriminative approach gains over the generative approach being able to better compensate the latent variables of speech data often lost in training data during the transformation from one physiological layer of abstraction to the next as discussed in section \ref{ASRMLP}.  This rationale is reinforced from the notion of deep learning defined in \cite{deng2014deep} as an attempt to learn patterns from data at multiple levels of abstraction. Thus while shallow machine learning models like hidden Markov models (HMMs) define latent variables for fixed layers of abstraction, deep machine learning models handle hidden/latent information for arbitrary layers of abstraction determined heuristically.  As deep learning are typically implemented using deep neural networks, this work applies deep recurrent neural networks as an end-to-end discriminative classifier, to speech recognition.  This is a so called "an end-to-end model" because it adopts the top-to-bottom machine learning approach. Unlike the typical generative classifiers that require sub-word acoustic models, the end-to-end models develop algorithms at higher levels of abstraction as well as the lower levels of abstraction.  In the case of the deep-speech model \citep{hannun2014first} used in this research, the levels of abstraction include sentence/phrase, words and character discrimination. A second advantage of the end-to-end model is that because the traditional generative models require various stages of modeling including an acoustic, language and lexicon, the end-to-end discriminating multiple levels of abstractions simultaneously only requires a single stage process, greatly reducing the amount of resources required for speech recognition.  From a low resource language perspective this is an attractive feature meaning that the model can be learned from an acoustic only source without the need of an acoustic model or a phonetic dictionary.  In theory this deep learning technique is sufficient in itself without a language model.  However, applying a language model was found to serve as a correction factor further improving recognition results citep{hannun2014deep}. 

## Low Resource Languages
A second challenge observed in complex machine learning models for both generative as well as discriminative learning models is the data intensive nature required for robust classification models. \cite{saon2015ibm} recommends around 2000 hours of transcribed speech data for a robust speech recognition system. As we cover in the next chapter, for new languages for which are low in training data such as transcribed speech, there are various strategies devised for low resource speech recognition. \cite{besacier2014automatic} outlines various matrices for benchmarking low resource languages.  From the generative speech model interest perspective,  reference is made to languages having less than ideal data in transcribed speech, phonetic dictionary and a text corpus for language modelling.  For end-to-end speech recognition models interests, the data relevant for low resource evaluation is the transcribed speech and a text corpus for language modelling.  It is worth noting that it was observed \citep{besacier2014automatic} that speaker-base atimes doesn't affect a language resource status of a language and was often observed that large speaker bases could in fact lack language/speech recognition resources and that some languages having small speaker bases did in fact have sufficient language/ speech recognition resources.

Speech recognition methods looked at in this work was motivated by the Wakirike language discussed in the next section, which is a low resource language by definition.  Thus this research looked at low research language modeling for the Wakirike language from a corpus of wakirike text available for analysis.  However, due to the insufficiency of transcribed speech for the Wakirike langauge, English language was substituted and used as a control variable to study low resource effects of a language when exposed to speech models developed in this work.

## The Wakirike Language
The Wakirike municipality is a fishing community comprising 13 districts in the Niger Delta area of the country of Nigeria in the West African region of the continent of Africa.  Okrika migrants settled at the Okrika mainland between AD860 at the earliest AD1515.  Earliest settlers migrated from Central and Western Niger Delta.  When the second set of settlers met the first set of settlers they exclaimed “we are not different” or “Wakirike” \citep{wakirike}.  Although the population of the Wakirike community from a 1995 report \citep{ethnologue} is about 248,000, the speaker base is  much less than that.  The language is classified as Niger-Congo and Ijoid languages.  The writing orthography is latin and the language status is 5 (developing) \citep{ethnologue}.  This means that although the language is not yet an endangered language, it still isn't thriving and it is being passed on to the next generation at a limited rate.

The Wakirike language was the focus for this research.  And End-to-end deep neural network language model was built for the Wakirike language based on the availability of the new testament bible printed edition that was available for processing.  The corpus utilized for this thesis work was about 9,000 words.

Due to limiations in transcribed speech for the Wakirike language, English was substituted and used for the final speech model.  The English language was used as a control variable to measure accuracy of speech recognition for differing spans of speech data being validated against on algorithms developed in this research.

## Thesis Outline
The outline of this report follows the development of an end-to-end speech recogniser and develops the theory based on the building blocks of the final system.  Chapter two introduces the speech recognition pipeline and the generative speech model.  Chapter two outlines the weaknesses in the generative model and describes some of the machine learning techniques applied to improve speech recognition performance. 

Low speech recognition are reviewed and the relevance of this study is also highlighted.  Chapter three describes Recurrent neural networks beginning from multi-layer perceptrons and probabilistic sequence models.  Specialised recurrent neural networks, long short-term memory (LSTM) networks and the Gated Recurrent Units (GRU) used to develop the language model for the Wakirike language.

Chapter Four explains the wavelet theorem as well as the deep scattering spectrum. The chapter develops the theory from fourier transform and details the the significance of using the scattering transform as a feature selection mechanism for low resource recognition.  

Chapters five and six is a description of the models developed by this thesis and details the experiment setup along with the results obtained. Chapters seven is a discussioin of the result and chapter 8 are the recommendations for further study. 

# Low Resource Speech Models, End-to-end models and the scattering network
The speech recogniser developed in this thesis is based on an end-to-end discriminative deep recurrent neural network.  Two models were developed.  The first model is a Gated-Recurrent-Unit Recurrent Neural network (GRU-RNN) was used to develop a character-based language model, while the second recurrent neural network is a Bi-Directional Recurrent neural Network (BiRNN) used as an end-to-end speech model capable of generating word sequences based on learned character sequence outputs.  This chapter describes the transition from generative speech models to these discriminative end-to-end recurrent neural network models.  Low speech recognition strategies are also introduced and the contribution to knowledge gained by introducing deep scattering features as inputs to the biRNN speech model is brought to light.

## Speech Recognition Overview
Computer speech recognition takes raw audio speech and converts it into a sequence of symbols.  This can be considered as an analog to digital conversion.  The way this conversion is done is by breaking up the audio sequence into very small packets referred to as frames and developing discriminating parameters or features for each frame. Then using the vector of features as input to the speech recogniser.  

The statistical formulation \citep{young2002htk} is as follows.  Given that each spoken word in the audio speech signal is represented as a vector sequence of observations defined in the set $$\mathbf{O}$$. Therefore
\begin{equation}$$\mathbf{O}=\mathbf{o}_1,\mathbf{o}_2,\dots,\mathbf{o}_T$$
\label{eqn_2_1_obseq}
\end{equation}

At each discrete time $$t$$, we have an observation $$\mathbf{o}_t$$, which in itself is a vector in $$\mathbb{R}^D$$.  From the conditional probability in mathematics and statistics it can be formulated that certain words in from a finite dictionary are most probable given a sequence of observations. That is:
\begin{equation}$$arg\max_t\{P(w_i|\mathbf{O})\}$$
\label{eqn_2_2_srgen}
\end{equation}

As we describe in the next section on speech recognition challenges, there is no straightforward analsysis of of $$P(w_i|\mathbf{O})$$.  The divide and conquer strategy therefore employed uses Bayes formulation to simplify the problem.  Accordingly, the argument that maximises the probability of an audio sequence given a particular word multiplied by the probability of that word is equivalent to the original posterior probability required to solve the isolated word recognition problem. This is summarised by the following equation
\begin{equation}$$P(w_i|\mathbf{O})=\frac{P(\mathbf{O}|w_i)P(w_i)}{P(\mathbf{O}}$$
\label{eqn_2_3_bayes_sr}
\end{equation}

That is, according to Bayes’ rule, the posterior probability is obtained by multiplying a certain likelihood probability by a prior probability.  The likelihood in this case, $$P(\mathbf{O}|w_i)$$, is obtained from a Hidden Markov Model (HMM) parametric model such that rather than estimating the observation densities in the likelihood probability, these are obtained by estimating the parameters of the HMM model.  The HMM model explained in the next section gives a statistical representation of the latent variables of speech.

### HMM-based Generative speech model
A HMM represents a finite state machine where a process transits a sequence of states from a set of fixed states. The overall sequence of transitions will have a start state, an end state and a finite number of intermediate states all within the set of finite states.  For each state transition emits an output observation that represents the current internal state of the system.
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Generative HMM model")
In an HMM represented in figure \ref{fig_2_1_hmm}
 there are two important probabilities.  The first is the state transition probability given by aij this is the probability to move from state i to state j.  The second probability bj is the probability that an output probability when a state emits an observation.

Given that X represents the sequence of states transitioned by a process a HMM the joint probability of X and the output probabilities given the HMM is given as the following representation:


### Challenges of Speech Recognition
   The realised symbol is assumed to have a one to one mapping with the segmented raw audio speech. However, the difficulty in computer speech recognition is the fact that there is significant amount of variation in speech that would make it practically intractable to establish a direct mapping from segmented raw speech audio to a sequence of static symbols. The phenomena known as coarticulation has it that there are several different symbols having a mapping to a single waveform of speech in addition to several other varying factors including the speaker mood, gender, age, the speech transducing medium, the room acoustics. Et cetera.

Another challenge faced by automated speech recognisers is the fact that the boundaries of the words is not apparent from the raw speech waveform. A third problem that immediately arises from the second is the fact that the words from the speech may not strictly follow the words in the selected vocabulary database.  Such occurrence in speech recognition research is referred to as out of vocabulary (OOV) terms.  It is reasonable to approach these challenges using a divide and conquer strategy.  In this case, the first step in this case would be to create assumption that somehow word boundaries can be determined.  This first step in speech recognition is referred to as the isolated word recognition case.

### Challenges of low speech recognition
Speech recognition for low resource languages poses another distinct set of challenges.  In chapter one, low resource languages were described to be languages lacking in resources required for adequate machine learning of models required for generative speech models.  These resources are described basically as a text corpus for language modelling and a phonetic dictionary and transcribed audio speech for acoustic modelling.

## Low Resource Speech Recognition
### Low Resource language modelling
Within the statistical ASR paradigm  and consequently linguistic domain which it imitates, we derive the unit of structure and the unit function.  The unit of structure for both the linguistic and ASR domain is the phoneme while the unit of function within both domains are the words. In addition the ASR paradigm takes this a step further and defines the unit of operation which consists of one or more words known as the speech utterance.  Note that the unit of function defines higher order linguistic units that includes syntax and semantics and some authors add pragmatics into the mix (Juang & Furui, 2000).

In acoustic model a lower level of syntax processing is realised.  In language modelling higher order syntax and low level semantics is being applied.  This report obtains that higher order semantics and pragmatics is beyond the scope of speech recognition and in the realm of speech understanding.

In the framework of language modelling, Juang and Furui(2000), Young(1996) evaluate the development. Language modelling formulate rules that predict linguistic events and can be modeled in terms discrete density P(W), where  W=(w1, w2,..., wL) is a word sequence. The density function P(W) assigns a probability to a particular word sequence W.  This value is determines how likely the word is to appear in an utterance. A sentence with words appearing in a grammatically correct manner is more likely to be spoken than a sentence with words mixed up in an ungrammatical manner, and, therefore, is assigned a higher probability. The order of words therefore reflect the language structure, rules, and convention in a probabilistic way. Statistical language modeling therefore, is an estimate for P(W)￼ from a given set of sentences, or corpus.

The prior probability of a word sequence $$\mathbf{w}=w_1,\dots,w_k$$ required in equation (2.2) is given by
\begin{equation}$$P(\mathbf{w})=\prod_{k=1}^KP(w_k|w_{k-1},\dots,w_1)$$
\label{eqn_c2_lm01}
\end{equation}

The N-gram model is formed by conditioning of the word history in equation \ref{eqn_c2_lm01}.  This therefore becomes
\begin{equation}$$P(\mathbf{w})=\prod_{k=1}^KP(w_k|w_{k-1},w_{k-2},\dots,w_{k-N+1})$$
\label{eqn_c2_lm02}
\end{equation}

N is typically in the range of 2-4.

N-gram probabilities are estimated from training corpus by counting N-gram occurrences.  This is plugged into maximum likelihood (ML) parameter estimate. For example, Given that N=3 then the probability that three words occurred is assuming $$C(w_{k-2}w_{k-1}w_k)$$ is the number of occurrences of the three words $$C(w_{k-2}w_{k-1})$$ is the count for $$w_{k-2}w_{k-1}w_k$$ then
\begin{equation}
$$P(w_k|w_{k-1},w_{k-2}\approx\frac{C(w_{k-2}w_{k-1}w_k}{C(w_{k-2}w_{k-1})}$$
\label{eqn_c2_lm03}
\end{equation}

The major problem with maximum likelihood estimation scheme is data sparsity. This can be tackled by a combination of smoothing techniques involving discounting and backing-off.  The alternative approach to robust language modelling is the so-called class based models (Brown, et al., 1992; Kuhn et al., 1998) in which data sparsity is not so much an issue. Given that for every word $$w_k$$, there is a corresponding class $$c_k$$, then,
\begin{equation}
$$P(\mathbf{w})\prod_{k=1}^KP(w_k|c_k)p(c_k|c_{k-1},\dots,c_{k-N+1})$$
\label{eqn_c2_lm04}
\end{equation}

The measure of a statistical language model is the entropy or perplexity. This value measures the complexity of a language that the language model is designed to represent (Jelinek, 1997). In practice, the entropy of a language with an N-gram language model PN(W) is measured via a set of sentences and is defined as
\begin{equation}$$H=\sum_{\mathbf{W}\in\Omega}P_N(\mathbf{W})$$
\label{eqn_c2_lm05}
\end{equation}

where Ω is a set of sentences of the language. The perplexity, which is interpreted as the average word-branching factor, is defined as
\begin{equation}$$B=2^H$$
\label{eqn_c2_lm06}
\end{equation}

For the neural network implementations so far seen, a largeamount of data is required due to the nature of words to havelarge vocabularies,  even for medium-scale speech recognitionapplications.   Yoon  Kim  et.   al.   [12]  on  the  other  hand  tooka  different  approach  to  language  modelling  taking  advantageof the long-term sequence memory of long-short-term memorycell recurrent neural network (LSTM-RNN) to rather model alanguage based on characters rather than on words. This greatlyreduced the number of parameters involved and therefore thecomplexity of implementation.  This method is particularly ofinterest to this article and forms the basis of the implementa-tion described in this article due to the low resource constraintsimposed when using a character-level language model.

Other  low  resource  language  modelling  strategies  em-ployed for the purpose of speech recognition was demonstratedby [13]. The language model developed in that work was basedon  phrase-level  linguistic  mapping  from  a  high  resource  lan-guage to a low resource language using a probabilistic modelimplemented using a weighted finite state transducer (WFST).This method uses WFST rather than a neural network due toscarcity of training data required to develop a neural network.However, it did not gain from the high nonlinearity ability of aneural network model to discover hidden patterns in data, beinga shallower machine learning architecture.The method employed in this article uses a character-basedNeural  network  language  model  that  employs  an  LSTM  net-work similar to that of [12] on the Okrika language which isa  low  resource  language  bearing  in  mind  that  the  characterlevel network will reduce the number of parameters required fortraining just enough to develop a working language model forthe purpose of speech recognition.  The description of the dataand procedure used to develop the language model is discussedin the next section.

The method employed in this article uses a character-basedNeural  network  language  model  that  employs  an  LSTM  net-work similar to that of [12] on the Okrika language which isa  low  resource  language  bearing  in  mind  that  the  characterlevel network will reduce the number of parameters required fortraining just enough to develop a working language model forthe purpose of speech recognition.  The description of the dataand procedure used to develop the language model is discussedin the next section

### Low Resource Acoustic modelling
Sub-space GMM for multi-lingual purpose

Sub-space Gaussian Mixture Models (SGMMs) has been shown to be suitable for cross-lingual modeling without explicit mapping between phone units in different languages.
In an SGMM, emission densities of a hidden Markov Model (HMM) are modeled as mixtures of Gaussians, whose parameters are factorized into a globally-shared set that does not depend on the HMM states, and a state specific set.
The global parameters may be thought of as a model for the overall acoustic space, while the state-specific parameters provide the correspondence between different regions of the acoustic space and individual speech sounds.
The decoupling  of two aspects of speech modeling that makes SGMM suitable for different languages.


DNN for multilingual speech recognition
Using layer wise pretraining of stacked Restricted Boltzmann Machines (RBMs) is shown to be insensitive to the choice of languages analogous to global parameters of SGMMs.
Using a network whose output layer corresponds to context-dependent phone states of a language, by borrowing the hidden layers and fine-tune the network to a new language.
The new outputs are scaled likelihood estimates for states of an HMM in a DNN-HMM recognition setup.
Used a 7-layer network without a bottleneck layer where the network outputs correspond to triphone states trained on MFCC features. Each layer contained about 2000 neurons.

## Groundwork for low resource end-to-end speech modelling
The underpinning notion of this work is firstly a departure from the bottom-to-top baggage that comes as a bye-product of the generative process sponsored by the HMM-based speech models so that we can gain from simplifying the speech pipeline from acoustic, language and phonetic model to just a speech model that approximates the same process.  Secondly, the model developed seeks to overcome the data intensity barrier and was seen to achieve measurable results for GRU RNN language models.  Therefore adopting the same character-based strategy, this research performed experiments using the character-based bi-directional recurrent neural networks (BiRNN) as used in \cite{hannun2014first}.  However, the authors refer to BiRNNs as also being very data intensive.  The next paragraphs introduce deepspeech BiRNNs and two strategies for reducing the data intensity drawback as related to low speech resource recognition.

### Deep speech
HMM free approach to SR
CTC loss function - maximises the likelihood of an output sequence by summing over all possible input-output sequence alignments.
CER was 10% on WSJ.
Integrated first-pass language model addition.
BiRNNs are less complex than LSTMs yet overcome vanishing gradient problem

### Speech Recognition on a low budget
Transfer learning based on model adaptation for training ASR models under constrained
GPU memory
Throughput
Training data
Model introspection (freezing) revealed that small adaptation to network weights were sufficient for good performance, especially for inner layers.
Related work
Heterogeneous transfer learning
Wang and Zheng, 2015
Chen and Mak 2015
Knill et al 2014
Heterogeneous transfer learning requires large amount of data
Heigold et al., 2013
Wang and Zheng (2015) demonstrates what amount of data is required for effective retraining.  This model was adapted in this work as follows
Train a model with one or  more languages
Retrain all or parts of it on another language which was unseen during the first training round.
Parameters learned from first language serve as starting point.
This was also done by Vu & Schultz (2013 by first learning an MLP from multiple languages with relative abundant data. In this work however, compressed bottle neck features (Grezl and Fousek, 2008) weren’t used.
In addition layer freezing was used as a low resource saving strategy in figure 1.

Model Architecture
Amodei et al (2015) CNN training using many GPUs was found to be complex having many an RNN stacked with many units
Collobert (2016) proposed  Wav2Letter architecture which relies entirely on its loss function to handle aligning  the audio and transcriptions while the network consists only of convolutional units.  This model did not sacrifice accuracy while improving speed of training.
Although the optimiser used in Collbert et al’s model, wasn’t specified, Adam(Kingma and Ba, 2014) achieved good convergence results in this study.
Relus have been shown to work well for acoustic models (Maas et al., 2013)
Weight’s were initialised Xavier uniformly Glorot and Bengio (2010).
Decoding was via beam search based on KenLM (Heafield et al., 2013).
Tensorflow was implemented based on (Graves, 2012) and training on LibriSpeech (Panayotov et al., 2015)
CNN was done using Keras (Chollet, 2015) and introspection using Numpy and tensorflow (Vander walt et al., 2011 and Abadi et al., 2015).
Loss function in Collbert et al. (2016) was using AutoSegCriterion as opposed to CTC loss function in Graves et al (2006)
Dataset - 1000 hours of read speech sampled at 16kHz. German data was 383 hours
Features - MFCCs
Preprocessing
Since the English model was trained on data with a sampling rate of 16 kHz, the German speech data needed to be brought into the same format so that the convolutional filters could operate on the same timescale. To this end, all data was resampled to 16 kHz. Preprocessing was done using librosa (McFee et al., 2015) and consisted of applying a Short-time Fourier transform (STFT) to obtain power level spectrum features from the raw audio as described in Colbert 2016.
After that, spectrum features were mel-scaled and then directly fed into the CNN. Originally, the parameters were set to window length w = 25ms, stride s = 10ms and number of components n=257. 
We adapted the window length to wnew = 32ms which equals a Fourier transform window of 512 samples, in order to follow the convention of using power-of-two window sizes.The stride was set to snew =8ms in order to achieve 75% overlap of successive frames. We observed that n = 257 results in many of the components being 0 due to the limited window length. We therefore decreased the parameter to nnew=128. After the generation of the spectrograms, we normalized them to mean 0 and standard deviation 1 per input sequence.
Any individual recordings in the German corpora longer than 35 seconds were removed due to GPU memory limitations.

### Adding a Scattering layer
Many speech and music classifiers use mel-frequency cepstral coefficients (MFCCs), which are cosine transforms of mel-frequency spectral coefficients (MFSCs).
Over a fixed time interval, MFSCs measure the signal frequency energy over mel-frequency intervals of constant-Q bandwidth.
 As a result, they lose information on signal structures that are non-stationary on this time interval.
To minimize this loss, short time windows of 23 ms are used in most applications since at this resolution most signals are locally stationary.
The characterization of audio properties on larger time scales is then done by aggregating MFSC coefficients in time, with multiple ad-hoc methods such as Delta-MFCC [5] or MFCC segments [1].
Paper shows that the non-stationary behavior lost by MFSC coefficients is captured by a scattering transform which computes multiscale co-occurrence coefficients
A scattering representation includes MFSC-like measurements together with higher-order co-occurrence coefficients that can characterize audio information over much longer time intervals, up to several seconds. This yields efficient representations for audio classification. 
Paper shows information lost by spectral energy measurements can be recovered by a scattering operator introduced in [8]
Co-occurrence coefficients can be calculated by cascading wavelet filter banks and rectifiers calculated with modulus operators.
A scattering transform has strong similarities with auditory physiological models based on cascades of constant-Q filter banks and rectifiers [4, 10]
It is shown that second-order co-occurrence coefficients carry an important part of the signal information

# RNN
## Sequential models
## Neural networks
## LSTM network

# Deep Scattering Network
## Fourier transform
## Mel filter banks
## Wavelet transform
The Fourier transform discussed in the previous section constitutes a valuable tool for the analysis of the frequency component of a signal.
## Deep scattering spectrum

# Wakirike Language Models
## Wakirike Language model
## Grapheme to phoneme model

# LSTM Speech Models
## Deep speech model
## CTC decoder
## DSS model

# Conclusion and Discussion

# Future Direction
## Pidgin English models
## OCR exploration
## GAN exploration

# Appendices
# References
references:bib.md

# Highland Scratchpad

<!--[Highland-ScratchPad-Start]
# Thesis notes
## Chapter 1
* Summary
* BiRNN, GRU, Scatnet references
* OOV ref?

## Chapter 2
* discriminative AM models
### Contribution to knowledge
1. BRNN simplifies processing
2. Scattering network increases feature representation for discrimination
### Methodology

## Chapter 3

[Highland-Bin-End]-->
