"""
Define constants.
"""
EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids
SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1,'O': 2, 'S-Nh': 3, 'S-Ns': 4, 'S-Ni': 5, 'B-Ni': 6, 'B-Ns': 7, 'E-Ni': 8, 'B-Nh': 9, 'I-Ni': 10, 'E-Ns': 11, 'I-Ns': 12}

OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1,'O': 2, 'S-Nh': 3, 'S-Ns': 4, 'S-Ni': 5, 'B-Ni': 6, 'B-Ns': 7, 'B-Nh': 8, 'E-Ns': 9, 'I-Ns': 10, 'I-Nh': 11, 'E-Ni': 12, 'I-Ni': 13, 'E-Nh': 14}

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1,'O': 2, 'S-Nh': 3, 'S-Ns': 4, 'B-Ni': 5, 'E-Ni': 6, 'S-Ni': 7, 'I-Ni': 8, 'B-Ns': 9, 'I-Ns': 10, 'E-Ns': 11, 'B-Nh': 12, 'E-Nh': 13, 'I-Nh': 14}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1,'nt': 2, 'n': 3, 'v': 4, 'wp': 5, 'nd': 6, 'a': 7, 'd': 8, 'c': 9, 'r': 10, 'u': 11, 'nh': 12, 'm': 13, 'p': 14, 'nz': 15, 'b': 16, 'q': 17, 'ns': 18, 'i': 19, 'ws': 20, 'j': 21, 'ni': 22, 'nl': 23, 'z': 24, 'h': 25, 'k': 26, 'o': 27, 'e': 28}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1,'Time': 2, 'Agt': 3, 'Root': 4, 'mPunc': 5, 'Loc': 6, 'mRang': 7, 'Mann': 8, 'mNeg': 9, 'mMod': 10, 'eSucc': 11, 'Pat': 12, 'mConj': 13, 'mTime': 14, 'Dir': 15, 'ePurp': 16, 'Nmod': 17, 'Exp': 18, 'Quan': 19, 'dCont': 20, 'mVain': 21, 'mDegr': 22, 'mPrep': 23, 'Datv': 24, 'Feat': 25, 'eCoo': 26, 'Poss': 27, 'mAux': 28, 'Cont': 29, 'Tmod': 30, 'Orig': 31, 'Qp': 32, 'mFreq': 33, 'Aft': 34, 'eResu': 35, 'eAdvt': 36, 'eEqu': 37, 'Tool': 38, 'dClas': 39, 'rProd': 40, 'Clas': 41, 'mTone': 42, 'Seq': 43, 'eCau': 44, 'dFeat': 45, 'mDir': 46, 'dExp': 47, 'eProg': 48, 'dTime': 49, 'Prod': 50, 'Sco': 51, 'rTime': 52, 'mPars': 53, 'rExp': 54, 'Accd': 55, 'Belg': 56, 'Reas': 57, 'Proc': 58, 'mMaj': 59, 'eConc': 60, 'dMann': 61, 'Host': 62, 'ePrec': 63, 'rAgt': 64, 'eRect': 65, 'dReas': 66, 'Cons': 67, 'rPat': 68, 'rLoc': 69, 'rCont': 70, 'dCons': 71, 'Freq': 72, 'eSelt': 73, 'eSupp': 74, 'Matl': 75, 'Comp': 76, 'mSepa': 77, 'rDatv': 78, 'dPoss': 79, 'dDatv': 80, 'rDir': 81, 'rPoss': 82, 'dDir': 83, 'eCond': 84, 'eMetd': 85, 'rBelg': 86, 'eInf': 87, 'dLoc': 88}

NEGATIVE_LABEL = 'no_relation'

LABEL_TO_ID = {'no_relation':0,'组织关系': 1, '灾害/意外': 2, '产品行为': 3, '司法行为': 4, '交往': 5, '财经/交易': 6, '人生': 7, '组织行为': 8, '竞赛行为': 9}

INFINITY_NUMBER = 1e12