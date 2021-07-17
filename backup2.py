from tkinter import *
from tkinter import filedialog
import numpy as np
from Source.summarizer import FrequencySummarizer

from nltk import sent_tokenize
from nltk.tokenize import word_tokenize

class FuzzySummarizer():

    @staticmethod
    def Fuzzy1(text1,numberoflines,title1):
            from operator import itemgetter
            
            with open(text1, 'r') as myfile:
                text=myfile.read()
            
            #print("Original Content: \n" ,text)
            #print("\n")
            sents = sent_tokenize(text)
            n2 = len(sents)
            
            def rescale(values, new_min = 0, new_max = 100):
                output = []
                old_min, old_max = min(values), max(values)
                for v in values:
                    new_v = (new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min
                    output.append(new_v)
                return output
            
            
            
            fs = FrequencySummarizer()
            
            
            
            rank=[]
            #for s in fs.summarize(text, 10):
            	#print ('*',s)
            
            for fq in fs.rankscore(text,numberoflines):
                #print(fq)
                for key, value in fq.items():
                    rank.append(value)
                    #print(rank)
                
            #print(rank,"\n")
            rescaled = rescale(rank)
            #print(rescaled,"\n")
            finalrank = []
            
            for i in range (0,n2):
                finalrank.append((i,rescaled[i]))
            
            #print(finalrank)
            #print(type(finalrank))
            finals2 = []
            finals2 = sorted(finalrank, key=itemgetter(1), reverse=True)
            #print (finals2,"\n")
            
            
            # # Sentence segmentation
            
            
            sentences=(sent_tokenize(text))
            #print("Sentences:",sentences)
            #print("\n")
            #print(len(sentences))
            
            """
            emptyarray= np.empty((len(sentences),1,3),dtype=object)
            for s in range(len(sentences)):
                emptyarray[s][0][0] = sentences[s]
                emptyarray[s][0][1] = s
            """
            
            # # Tokenization, Stop word removal
            
            import nltk
            from string import punctuation
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
              
            # # Sentence Position Feature
            
            
            def position(l):
                return [index for index, value in enumerate(sentences)]
            
            sent_position= (position(sentences))
            num_sent=len(sent_position)
            position = []
            position_rbm = []
            sent_pos1_rbm = 1
            sent_pos1 = 100
            position.append(sent_pos1)
            position_rbm.append(sent_pos1_rbm)
            for x in range(1,num_sent-1):
                s_p= ((num_sent-x)/num_sent)*100
                position.append(s_p)
                s_p_rbm = (num_sent-x)/num_sent
                position_rbm.append(s_p_rbm)
                
            sent_pos2 = 100
            sent_pos2_rbm = 1
            position.append(sent_pos2)
            #Normalized position values
            position_rbm.append(sent_pos2_rbm)
            
            # # Sentence to Sentence Cohesion using Cosine Similarity
            
            import re, math
            from collections import Counter
            
            WORD = re.compile(r'\w+')
            
            def get_cosine(vec1, vec2):
                 intersection = set(vec1.keys()) & set(vec2.keys())
                 numerator = sum([vec1[x] * vec2[x] for x in intersection])
            
                 sum1 = sum([vec1[x]**2 for x in vec1.keys()])
                 sum2 = sum([vec2[x]**2 for x in vec2.keys()])
                 denominator = math.sqrt(sum1) * math.sqrt(sum2)
            
                 if not denominator:
                    return 0.0
                 else:
                    return float(numerator) / denominator
            
            def text_to_vector(text):
                 words = WORD.findall(text)
                 return Counter(words)
            
            sim_score = []
            
            for i in range(0,len(sentences)):
                sim_score.append([])
            
            for i in range(0,len(sentences)):
                for j in range(0,len(sentences)):
                    if(j!=i):
                        vector1 = text_to_vector(sentences[i])
                        vector2 = text_to_vector(sentences[j])
                    
                        cosine = get_cosine(vector1, vector2)
                        sim_score[i].append((cosine))
            
            
            cos_score = []
            cos_score_norm = []
            
            for i in range(0,len(sentences)):
                cos_score.append(sum(sim_score[i]))
                
            cos_score_norm = rescale(cos_score)
            
            # # Sentence length feature
            sent_word=[]
            for u in range(len(sentences)):
                sent_split1=[w.lower() for w in sentences[u].split(" ")]
                sent_split=[w for w in sent_split1 if w not in stop_words and w not in punctuation and not w.isdigit()]
                a=(len(sent_split))
                sent_word.append(a)
            
            longest_sent=max(sent_word)
            sent_length=[]
            sent_length_rbm=[]
            for x in sent_word:
                sent_length.append((x/longest_sent)*100)
                sent_length_rbm.append(x/longest_sent)
            
            # # Numeric token Feature
            
            import re
            num_word=[]
            numeric_token=[]
            numeric_token_rbm=[]
            for u in range(len(sentences)):
                sent_split4=sentences[u].split(" ")
                e=re.findall("\d+",sentences[u])
                noofwords=(len(e))
                num_word.append(noofwords)
                numeric_token.append((num_word[u]/sent_word[u])*100)
                numeric_token_rbm.append(num_word[u]/sent_word[u])
            
            # # Thematic words feature
            
            from rake_nltk import Rake
            r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
            keywords=[]
            # If you want to provide your own set of stop words and punctuations to
            # r = Rake(<list of stopwords>, <string of puntuations to ignore>)
            for s in sentences:
                r.extract_keywords_from_text(s)
                key=list(r.get_ranked_phrases())
                keywords.append(key)
            #print(keywords)
            l_keywords=[]
            for s in keywords:
                leng=len(s)
                l_keywords.append(leng)
            total_keywords=max(l_keywords)
            #print(total_keywords)
            thematic_keyword= []
            thematic_keyword_rbm= []
            for x in l_keywords:
                thematic_keyword.append((x/total_keywords)*100)
                thematic_keyword_rbm.append(x/total_keywords)
            
            # # proper noun feature
            
            from collections import Counter
            pncounts = []
            pncounts_rbm = []
            for sentence in sentences:
                tagged=nltk.pos_tag(nltk.word_tokenize(str(sentence)))
                counts = Counter(tag for word,tag in tagged if tag.startswith('NNP') or tag.startswith('NNPS'))
                f=sum(counts.values())
                pncounts.append(f)
                pncounts_rbm.append(f)
            pnounscore=[(int(o) / int(p))*100 for o,p in zip(pncounts, sent_word)]
            pnounscore_rbm=[int(o) / int(p) for o,p in zip(pncounts_rbm, sent_word)]
            
            # # Similarity to Title
            sents = []
            
            for sentence in sentences:
                sents.append(sentence.lower())
            
            #print(sents,"\n\n")
            triallist = []
            trial1= []
            
            for i in range(0,len(sents)):
                trial = nltk.word_tokenize(str(sents[i]))
                trial1 = [w for w in trial if not w in stop_words and w not in punctuation]
                triallist.append(trial1)
                
            #print(triallist)
            
            title = title1
            titlewords = (nltk.word_tokenize(str(title)))
            
            filtered_title = []
            filtered_title = [w.lower() for w in titlewords if not w in stop_words]
            
            #print("TITLE: ",filtered_title)
            title_score = []
            title_score_rbm= []
            wordsintitle = len(filtered_title)
            
            for i in range(0,len(sents)):
                p = set(filtered_title)&set(triallist[i])
                title_score.append((len(p)/wordsintitle)*100)
                title_score_rbm.append(len(p)/wordsintitle)
            
            
            # In[2]:
            
            import skfuzzy as fuzz
            from skfuzzy import control as ctrl
            
            # New Antecedent/Consequent objects hold universe variables and membership
            # functions
            position1 = ctrl.Antecedent(np.arange(0, 100, 10), 'position1')
            cos_score_sim = ctrl.Antecedent(np.arange(0, 100, 10), 'cos_score_sim')
            title_sim = ctrl.Antecedent(np.arange(0, 100, 10), 'title_sim')
            propernoun = ctrl.Antecedent(np.arange(0, 100, 10), 'propernoun')
            sentencelength = ctrl.Antecedent(np.arange(0, 100, 10), 'sentencelength')
            numtokens = ctrl.Antecedent(np.arange(0, 100, 10), 'numtokens')
            keywords = ctrl.Antecedent(np.arange(0, 100, 10), 'keywords')
            
            
            senten = ctrl.Consequent(np.arange(0, 100, 10), 'senten')
            
            position1.automf(3)
            title_sim.automf(3)
            cos_score_sim.automf(3)
            propernoun.automf(3)
            sentencelength.automf(3)
            numtokens.automf(3)
            keywords.automf(3)
            
            
            
            senten['bad'] = fuzz.trimf(senten.universe, [0, 0, 50])
            senten['avg'] = fuzz.trimf(senten.universe, [0, 50, 100])
            senten['good'] = fuzz.trimf(senten.universe, [50, 100, 100])
            
            
            rule1 = ctrl.Rule(title_sim['good'] | sentencelength['good'] | position1['good'] | cos_score_sim['poor'] | propernoun['good'] | keywords['good'] | numtokens['good'], senten['good'])
            rule2 = ctrl.Rule(title_sim['good'] | sentencelength['good'] | position1['good'] | cos_score_sim['poor'] | propernoun['poor'] | keywords['good'] | numtokens['good'], senten['good'])
            rule3 = ctrl.Rule(title_sim['good'] | sentencelength['good'] | position1['good'] | cos_score_sim['poor'] | propernoun['good'] | keywords['good'] | numtokens['good'], senten['good'])
            rule4 = ctrl.Rule(title_sim['good'] | sentencelength['poor'] | position1['poor'] | cos_score_sim['good'] | propernoun['poor'] | keywords['poor'] | numtokens['poor'], senten['bad'])
            rule5 = ctrl.Rule(title_sim['poor'] | sentencelength['good'] | position1['poor'] | cos_score_sim['good'] | propernoun['poor'] | keywords['poor'] | numtokens['poor'], senten['bad'])
            rule6 = ctrl.Rule(title_sim['poor'] | sentencelength['poor'] | position1['poor'] | cos_score_sim['good'] | propernoun['poor'] | keywords['poor'] | numtokens['poor'], senten['bad'])
            rule7 = ctrl.Rule(title_sim['good'] | sentencelength['poor'] | position1['good'] | cos_score_sim['average'] | propernoun['poor'] | keywords['poor'] | numtokens['poor'], senten['avg'])
            rule8 = ctrl.Rule(title_sim['good'] | sentencelength['good'] | position1['good'] | cos_score_sim['average'] | propernoun['poor'] | keywords['poor'] | numtokens['poor'], senten['avg'])
            rule9 = ctrl.Rule(title_sim['poor'] | sentencelength['poor'] | position1['poor'] | cos_score_sim['average'] | propernoun['poor'] | keywords['good'] | numtokens['good'], senten['avg'])
            rule10 = ctrl.Rule(position1['good'] & sentencelength['good'] & title_sim['good'], senten['good'])
            rule11 = ctrl.Rule(position1['poor'] & sentencelength['poor'] & title_sim['poor'], senten['bad'])
            rule12 = ctrl.Rule(cos_score_sim['good'], senten['bad'])
            rule13 = ctrl.Rule(cos_score_sim['poor'], senten['good'])
            rule14 = ctrl.Rule(title_sim['average'] | sentencelength['average'] | position1['average'] | cos_score_sim['average'] | propernoun['average'] | keywords['average'] | numtokens['average'], senten['avg'])
            rule15 = ctrl.Rule(title_sim['good'] & sentencelength['good'] & position1['good'] | cos_score_sim['poor'] | propernoun['poor'] | keywords['poor'] | numtokens['poor'], senten['good'])
            rule16 = ctrl.Rule(title_sim['good'] & sentencelength['average'] & position1['average'] | cos_score_sim['poor'] | propernoun['good'] | keywords['good'] | numtokens['good'], senten['good'])
            rule17 = ctrl.Rule(title_sim['average'] & sentencelength['good'] & position1['average'] | cos_score_sim['poor'] | propernoun['good'] | keywords['good'] | numtokens['good'], senten['good'])
            rule18 = ctrl.Rule(title_sim['average'] & sentencelength['average'] & position1['good'] | cos_score_sim['poor'] | propernoun['good'] | keywords['good'] | numtokens['good'], senten['good'])
            rule19 = ctrl.Rule(title_sim['good'] & sentencelength['good'] & position1['good'] | cos_score_sim['poor'] | propernoun['poor'] | keywords['poor'] | numtokens['poor'], senten['avg'])
            rule20 = ctrl.Rule(title_sim['good'] & sentencelength['average'] & position1['average'] | cos_score_sim['poor'] | propernoun['poor'] | keywords['poor'] | numtokens['poor'], senten['avg'])
            rule21 = ctrl.Rule(title_sim['average'] & sentencelength['good'] & position1['average'] | cos_score_sim['poor'] | propernoun['poor'] | keywords['poor'] | numtokens['poor'], senten['avg'])
            rule22 = ctrl.Rule(title_sim['average'] & sentencelength['average'] & position1['good'] | cos_score_sim['poor'] | propernoun['poor'] | keywords['poor'] | numtokens['poor'], senten['avg'])
            rule23 = ctrl.Rule(title_sim['good'], senten['avg'])
            rule24 = ctrl.Rule(sentencelength['good'], senten['avg'])
            rule25 = ctrl.Rule(position1['good'], senten['avg'])
            rule26 = ctrl.Rule(propernoun['good'] & keywords['good'] & numtokens['good'], senten['avg'])
            rule27 = ctrl.Rule(propernoun['average'] & keywords['average'] & numtokens['average'], senten['bad'])
            
            
            sent_ctrl = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17,rule18,rule19,rule20,rule21,rule22,rule23,rule24,rule25,rule26,rule27])
            Sent = ctrl.ControlSystemSimulation(sent_ctrl)
            fuzzemptyarr= np.empty((20,1,2), dtype=object)
            t2=0
            summary2=[]
            finals=[]
            finalrescale=[]
            finals1=[]
            for s in range(len(sentences)):
                Sent.input['position1'] = int(position[s])
                Sent.input['cos_score_sim'] = int(cos_score_norm[s])
                Sent.input['title_sim'] = int(title_score[s])
                Sent.input['keywords'] = int(thematic_keyword[s])
                Sent.input['propernoun'] = int(pnounscore[s])
                Sent.input['sentencelength'] = int(sent_length[s])
                Sent.input['numtokens'] = int(numeric_token[s])
            #Sent.input['service'] = 2
                Sent.compute()
                finals.append(Sent.output['senten'] )
                
            
            finalrescale.append(rescale(finals))
            
            for i in range(0,n2) :
             finals1.append((i,finalrescale[0][i]))
            
            
            from operator import itemgetter
            finals1 = sorted(finals1, key=itemgetter(1), reverse=True)
            #print( finals1)
            
            #print(finals2)
            cut1=[]
            cut2=[]
            cut=[]
            ext=[]
            ext1=[]
            a=numberoflines
            print("\n")
            
            for i in range(0,a):
                cut1.append(finals1[i])
                cut2.append(finals2[i])
               
            cnt=0
            for i in range(0,a):
                flag=0
                for j in range(0,a):
                    if cut1[i][0]==cut2[j][0]:
                        cut.append(cut1[i]) 
                        flag=1    
                        cnt=cnt+1
                if flag==0:
                    ext.append(cut1[i])
                    
                    
                    
                    
                    
                    
                    
            for i in range(0,a):
                flag=0
                for j in range(0,a):
                    if cut2[i][0]==cut1[j][0]:
                        flag=1    
                       
                if flag==0:
                    ext.append(cut2[i])
            #print('cut.',cut)
            
            
            
            ext1 = sorted(ext, key=itemgetter(1), reverse=True)
            #print('\nExt',ext1)
            for i in range(0,a-cnt):
                cut.append(ext1[i])
                
                
            cut.sort()    
            #print('final',cut) 
            finalsummary=[]    
            for i in range(0,a):
              finalsummary.append(sentences[cut[i][0]])  
              
            #print('Final Summary: \n',*finalsummary,sep = "\n")
            #return finalsummary
            result = ""
            for x in finalsummary:
                result += x
            return result
       
