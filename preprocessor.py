import time
start = time.time()
import pandas as pd
import numpy as np
import re
import pickle
# import tensorflow as tf

with open('preprocessing_dicts/preprocessing_dicts.pkl', 'rb') as pickle_file:
    reqd_features = pickle.load(pickle_file)

# with open('feature_encoders/one_hot_encoder.pkl', 'rb') as pickle_file:
#     ohe = pickle.load(pickle_file)

# with open('scalers/rec_cnt_scaler.pkl', 'rb')as pickle_file:
#     rec_cnt_scaler = pickle.load(pickle_file)
# with open('scalers/sub_char_scaler.pkl', 'rb')as pickle_file:
#     sub_char_scaler = pickle.load(pickle_file)
# with open('scalers/content_char_scaler.pkl', 'rb')as pickle_file:
#     content_char_scaler = pickle.load(pickle_file)

# model = tf.keras.models.load_model("dl_model/ann_model.keras")


def return_dicts(email_text, reqd_features=reqd_features):
    data = dict()
    matches = re.findall(r'\b\w+(?:-\w+)*: ', email_text)
    matches = [match.strip() for match in matches]
    all_matches = ['From:']
    all_matches.extend(matches)
    split_text = re.split(r'\b\w+(?:-\w+)*: ', email_text)
    for i, j in zip(all_matches, split_text):
        for k in reqd_features:
            if k in i:
                data[i] = j
    return (
        data.get('To:', None),
        data.get('Subject:', None),
        data.get('Content-Type:', None),
        data.get('Precedence:', None),
        data.get('Content-Transfer-Encoding:', None),
        data.get('List-Unsubscribe:', None),
        data.get('List-Subscribe:', None),
        data.get('In-Reply-To:', None),
        data.get('wrote:', None),
    )


def column_dropper(data, col):
    transformed_df = data.copy()
    transformed_df.drop(columns=col, inplace=True)
    return transformed_df



email_text = '''
From ilug-admin@linux.ie Mon Jul 29 11:28:02 2002 Return-Path: <ilug-admin@linux.ie> Delivered-To: yyyy@localhost.netnoteinc.com Received: from localhost (localhost [127.0.0.1]) by phobos.labs.netnoteinc.com (Postfix) with ESMTP id A13D94414F for <jm@localhost>; Mon, 29 Jul 2002 06:25:11 -0400 (EDT) Received: from phobos [127.0.0.1] by localhost with IMAP (fetchmail-5.9.0) for jm@localhost (single-drop); Mon, 29 Jul 2002 11:25:11 +0100 (IST) Received: from lugh.tuatha.org (root@lugh.tuatha.org [194.125.145.45]) by dogma.slashnull.org (8.11.6/8.11.6) with ESMTP id g6RHn7i17130 for <jm-ilug@jmason.org>; Sat, 27 Jul 2002 18:49:07 +0100 Received: from lugh (root@localhost [127.0.0.1]) by lugh.tuatha.org (8.9.3/8.9.3) with ESMTP id SAA25016; Sat, 27 Jul 2002 18:45:03 +0100 X-Authentication-Warning: lugh.tuatha.org: Host root@localhost [127.0.0.1] claimed to be lugh Received: from mail1.mail.iol.ie (mail1.mail.iol.ie [194.125.2.192]) by lugh.tuatha.org (8.9.3/8.9.3) with ESMTP id SAA24977 for <ilug@linux.ie>; Sat, 27 Jul 2002 18:44:56 +0100 Received: from dialup125-a.ts551.cwt.esat.net ([193.203.140.125] helo=Hobbiton.cod.ie) by mail1.mail.iol.ie with esmtp (Exim 3.35 #1) id 17YVVF-0001W4-00 for ilug@linux.ie; Sat, 27 Jul 2002 18:37:18 +0100 Received: (from cdaly@localhost) by Hobbiton.cod.ie (8.11.6/8.9.3) id g6RDRoO04681 for ilug@linux.ie; Sat, 27 Jul 2002 14:27:50 +0100 Date: Sat, 27 Jul 2002 14:27:49 +0100 From: Conor Daly <conor.daly@oceanfree.net> To: ILUG main list <ilug@linux.ie> Subject: Re: [ILUG] Architecture crossover trouble w RH7.2 (solved) Message-Id: <20020727142749.B4438@Hobbiton.cod.ie> Mail-Followup-To: ILUG main list <ilug@linux.ie> References: <0D443C91DCE9CD40B1C795BA222A729E018854FA@milexc01.maxtor.com> MIME-Version: 1.0 Content-Type: text/plain; charset=us-ascii Content-Disposition: inline User-Agent: Mutt/1.2.5i In-Reply-To: <0D443C91DCE9CD40B1C795BA222A729E018854FA@milexc01.maxtor.com>; from conor_wynne@maxtor.com on Fri, Jul 26, 2002 at 03:56:22PM +0100 Sender: ilug-admin@linux.ie Errors-To: ilug-admin@linux.ie X-Mailman-Version: 1.1 Precedence: bulk List-Id: Irish Linux Users' Group <ilug.linux.ie> X-Beenthere: ilug@linux.ie On Fri, Jul 26, 2002 at 03:56:22PM +0100 or so it is rumoured hereabouts, Wynne, Conor thought: > Surely it would be faster to save you conf files, install it on the box > again, copy back you confs and voila. > All you car about are the confs as the boite has no DATA right? Yeah, but then I'd have to remember _exactly_ which confs I'd modified and they're not all in /etc either... > Thats what I would do, but you sysadmins have to make life as difficult & > complicated as possible ;--) Yup... In this case, I had two issues. 1. I mirrored the disk to give to someone else to work on but the box he has available has only a P1 or P2 processor. 2. My celeron box has been crashing the backup software so I wanted to try out the backup in a different box to make sure it's hardware related. Again, it's also an interesting exercise... > Have you thought about mirroring the system drives? Might save you serious > hassle down the line. Oh, I'm doing that too. This is going to Africa so I'm aiming for as robust as possible with belt, braces and probably an all-in-one jumpsuit! I'll be mirroring the disk but that is worth only so much (eg. lightning strike taking out the disk(s) or system compromise) I'm also going for a backup to CDR with an automated restore http://www.mondorescue.org . The admin out there wouldn't be able to build the system again if the mobo got fried and the replacement was the wrong arch but an i386 compatible install will mean just dropping in the HD and booting (ish)... Conor -- Conor Daly <conor.daly@oceanfree.net> Domestic Sysadmin :-) --------------------- Faenor.cod.ie 2:32pm up 64 days, 23:49, 0 users, load average: 0.00, 0.00, 0.00 Hobbiton.cod.ie 2:19pm up 7 days, 20:56, 1 user, load average: 0.05, 0.02, 0.00 -- Irish Linux Users' Group: ilug@linux.ie http://www.linux.ie/mailman/listinfo/ilug for (un)subscription information. List maintainer: listmaster@linux.ie
'''
end = round(time.time()-start, 2)

data = pd.DataFrame(pd.Series(email_text).apply(return_dicts).tolist(), columns=reqd_features)


data['Replied_mail'] = np.where(pd.isna(data['In-Reply-To:']), 0, 1)
data = column_dropper(data, 'In-Reply-To:')

data['Recievers_count'] = data['To:'].apply(lambda x: len(str(x).split('@')) - 1)
data = column_dropper(data, 'To:')
print(data)

data['Sub_Unsub_link'] = np.where(
    (pd.notna(data['List-Subscribe:']) & pd.notna(data['List-Unsubscribe:'])), 1, 0)

data = column_dropper(data, ['List-Subscribe:','List-Unsubscribe:'])

# def add_prec_content(row):
#     try :
#         return  row[5:].strip()
#     except TypeError :
#         return ""

# data['Prec_content'] = data['Precedence:'].apply(add_prec_content)
# data

# def parse_text(data, col):
#         for index, text in enumerate(data[col]):
#                 try :
#                         soup = BeautifulSoup(data[col][index], 'lxml')
#                         parsed_text = soup.text.strip()
#                 except Exception as ex :
#                         parsed_text = ""
#                 finally :
#                         data[col][index] = parsed_text
#         return data

# data = parse_text(data, 'Content-Transfer-Encoding:')
# data = parse_text(data, 'Prec_content')
# data = parse_text(data, 'wrote:')

# def clean_text(text):
#     return re.sub(r'[^a-zA-Z0-9\s]', '', text)
# data['Wrote_content'] = data['wrote:'].apply(clean_text)

# data.drop(columns=['wrote:'], inplace=True)

# data['Content-Type:'] = data['Content-Type:'].apply(lambda x : str(x).lower())
# data['Content-Type:'] = data['Content-Type:'].apply(lambda x: x[5:].split(';')[0].strip() if ';' in str(x) else x)
# data['Content-Type:'] = data['Content-Type:'].apply(lambda x: 'html' if 'html' in str(x) else ('none' if 'none' in str(x) else 'plain'))

# encoded_content_type = ohe.transform(data[['Content-Type:']])
# encoded_df = pd.DataFrame(encoded_content_type, columns=ohe.get_feature_names_out(['Content-Type:']))
# data['Content_type_html'] = encoded_df['Content-Type:_html'].astype(int)
# data['Content_type_plain'] = encoded_df['Content-Type:_plain'].astype(int)
# data.drop(columns=['Content-Type:'], inplace=True)

# data = parse_text(data, 'Subject:')
# data['Subject'] = data['Subject:'].apply(lambda x: str(x).strip())
# data['Subject'] = data['Subject'].apply(lambda x: "blank" if x == "" else x)
# data['Full_content'] = data['Content-Transfer-Encoding:'].astype(str) +" "+ data['Prec_content'].astype(str) +" "+ data['Wrote_content'].astype(str)
# data.drop(columns=['Prec_content','Wrote_content','Content-Transfer-Encoding:', 'Subject:', 'text'], inplace=True)
# data['Full_content'] = data['Full_content'].apply(lambda x: str(x).strip())
# data['Full_content'] = data['Full_content'].apply(lambda x: "blank" if x == "" else x)
# data['content_char'] = data['Full_content'].apply(len)
# data['sub_char'] = data['Subject'].apply(len)

# recievers_cnt = rec_cnt_scaler.transform(data[['Recievers_count']])
# data['Recievers_count'] = recievers_cnt

# sub_char = sub_char_scaler.transform(data[['sub_char']])
# data['sub_char'] = sub_char

# cont_char = content_char_scaler.transform(data[['content_char']])
# data['content_char'] = cont_char

# ps = PorterStemmer()

# def preprocessing_text(text):
#     text = re.sub('[^a-zA-Z]', ' ', text).lower()
#     words = text.split()
#     words = [ps.stem(word) for word in words if word not in stopwords.words('english') ]
#     return ' '.join(words)

# preprocessed_df = data.copy()
# preprocessed_df['Subject'] = preprocessed_df['Subject'].apply(preprocessing_text)
# preprocessed_df['Subject'] = preprocessed_df['Subject'].apply(lambda x: "blank" if x.strip() == "" else x)
# preprocessed_df['Full_content'] = preprocessed_df['Full_content'].apply(preprocessing_text)
# preprocessed_df['Full_content'] = preprocessed_df['Full_content'].apply(lambda x: "blank" if x.strip() == "" else x)

# content_vectors = tfidf_content.transform(preprocessed_df['Full_content']).toarray()
# subject_vectors = tfidf_subject.transform(preprocessed_df['Subject']).toarray()

# vec_array = np.concatenate((content_vectors, subject_vectors), axis=1)
