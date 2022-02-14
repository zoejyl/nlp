import os,fnmatch
import numpy as np
import re

dataDir = '/Users/Zoe/Documents/University/2018WINTER/CSC401/A3/data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    n = len(r)
    m = len(h)
    R = np.zeros((n+1,m+1))
    B = np.zeros((n+1,m+1))
    for i in range(n+1):
        R[i][0] = i
    for j in range(m+1):
        R[0][j] = j
    for k in range(n+1):
        B[k][0] = 1
    for k in range(m+1):
        B[0][k] = 2
    B[0][0] = 0

    for i in range(1,n+1):
        for j in range(1,m+1):
            dele = R[i-1][j]+1
            sub = R[i-1][j-1] + (0 if r[i-1] == h[j-1] else 1)
            ins = R[i][j-1]+1
            R[i][j] = np.amin(np.array([dele,sub,ins]))
            # 1 for up
            # 2 for left
            # 3 for up-left
            if R[i][j] == dele :
                B[i][j] = 1
            elif R[i][j] == ins:
                B[i][j] = 2
            else:
                B[i][j] = 3
    WER = R[n][m]/n
    arrow = B[n][m]
    nS = 0
    nI = 0
    nD = 0 
    i = n
    j = m 
    while arrow != 0:
        if arrow == 1:
            nD += 1
            i -= 1
            arrow = B[i][j]
        elif arrow == 2:
            nI +=1
            j -= 1
            arrow = B[i][j]
        else:
            if r[i-1] != h[j-1] :
                nS += 1
            i -= 1
            j -= 1
            arrow = B[i][j]
    

    return (WER,nS,nI,nD)

def preprocess(sentence):
    punctuation = re.compile('[!\.\?@"#\$\(\)\*\+,-/:;<=>\\\[\]\^_`\{\}\|~\'%&]')
    tag1 = re.compile('\[.+\]')
    tag2 = re.compile('<.+>')
    temp = sentence.strip().split()
    temp[0] = ''
    temp[1] = '' 
    for i in range(len(temp)):
        if re.search(tag2,temp[i]):
            temp[i]= re.sub(tag2,'',temp[i])
    for i in range(len(temp)):
        if re.search(tag1,temp[i]):
            temp[i]= re.sub(tag1,'',temp[i])
    tempstring = ''
    for item in temp:
        tempstring += item.strip()+' '
    newsentence = tempstring.strip()

    if punctuation.finditer(newsentence):
        matchiter = punctuation.finditer(newsentence)
        for match in matchiter:
            punc = match.group(0)
            newsentence= newsentence.replace(punc,"")
    newsentence = newsentence.lower()
    newsentence  = re.sub(r'\s+',' ',newsentence)
    return newsentence

if __name__ == "__main__":

    googleWER = []
    kaldiWER = []
    a = Levenshtein("sitting","kitten")


    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            google_fullfile = os.path.join( dataDir, speaker, 'transcripts.Google.txt' )
            google_file = open(google_fullfile,'r')
            google_lines = google_file.readlines()
            transcript_fullfile = os.path.join( dataDir, speaker, 'transcripts.txt' )
            transcript_file = open(transcript_fullfile,'r')
            transcript_lines = transcript_file.readlines()
            Kaldi_fullfile = os.path.join( dataDir, speaker, 'transcripts.Kaldi.txt' )
            Kaldi_file = open(Kaldi_fullfile,'r')
            Kaldi_lines = Kaldi_file.readlines()

            if len(google_lines) == 0 or len(Kaldi_lines) == 0 or len(transcript_lines) == 0:
                continue

            for i in range(len(transcript_lines)):
                transcript_lines[i] = preprocess(transcript_lines[i])
            for i in range(len(google_lines)):
                google_lines[i] = preprocess(google_lines[i])
            for i in range(len(Kaldi_lines)):
                Kaldi_lines[i] = preprocess(Kaldi_lines[i])
            for i in range(len(transcript_lines)):
                wer_1 = Levenshtein(transcript_lines[i].split(),google_lines[i].split())
                googleWER.append(wer_1[0])
                wer_2 = Levenshtein(transcript_lines[i].split(),Kaldi_lines[i].split())
                kaldiWER.append(wer_2[0])
                print(speaker,"Google",i,wer_1[0],"S:",wer_1[1],"I:",wer_1[2],"D:",wer_1[3])
                print(speaker,"Kaldi",i,wer_2[0],"S:",wer_2[1],"I:",wer_2[2],"D:",wer_2[3])
    avgGoogle = np.mean(np.array(googleWER))
    avgKaldi = np.mean(np.array(kaldiWER))
    stdGoogle = np.std(np.array(googleWER))
    stdKaldi = np.std(np.array(kaldiWER))
    print(avgGoogle)
    print(avgKaldi)
    print(stdGoogle)
    print(stdKaldi)





            

