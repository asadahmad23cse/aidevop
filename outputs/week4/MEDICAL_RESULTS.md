# Week 4 Quantitative Evaluation Results

> Generated only from recorded benchmark artifacts. Automated and rule-based evidence scores are explicitly identified; they are not blind human judgments.

## Validity

- Complete comparable matrix: **True**
- Records: 75 / 75
- Models: codellama_7b, qwen25_coder_3b, starcoder2_3b
- Tasks: 25

## Model comparison

| Model | Correctness | Relevance | Hallucination rate | Fact coverage proxy | Retrieval hit@k | Abstention accuracy | Mean latency ms | Mean completion tokens | Ollama RSS MB | Ollama VRAM MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| codellama_7b | 0.180 | 0.720 | 0.444 | 0.327 | 1.000 | 0.000 | 42315.0 | 204.1 | 19.5 | 5710.1 |
| qwen25_coder_3b | 0.180 | 0.800 | 0.271 | 0.253 | 1.000 | 0.500 | 9257.0 | 78.8 | 35.6 | 2055.3 |
| starcoder2_3b | 0.580 | 0.800 | 0.111 | 0.620 | 1.000 | 0.000 | 19444.5 | 181.4 | 25.0 | 1789.0 |

## Evidence-based analysis

- Highest automated required-fact coverage: starcoder2_3b (0.620).
- Lowest unsupported-sentence proxy: starcoder2_3b (0.144).
- Lowest mean response latency: qwen25_coder_3b (9257.0 ms).
- Highest evidence-adjudicated correctness: starcoder2_3b (0.580).
- Highest evidence-adjudicated relevance: starcoder2_3b (0.800).
- Lowest evidence-adjudicated hallucination rate: starcoder2_3b (0.111).
- Lowest Ollama model allocation on the Intel Arc iGPU: starcoder2_3b (1789.0 MB).
- A quality-latency trade-off is observed: starcoder2_3b gains 0.367 fact-coverage points over qwen25_coder_3b, while taking 10187.5 ms more per response on average.

## Metric interpretation

- Correctness, relevance, and hallucination rate use the completed evidence-adjudication CSV; its deterministic rubric is recorded in every row and can be replaced by blind human review.
- Retrieval quality uses labelled relevant documents/files: Precision@k, Recall@k, MRR, Hit@k, and nDCG@k.
- Test-pass rate is N/A because these suites ask QA/code-understanding questions and do not execute model-generated code.
- Latency is wall-clock request time after one warm-up; token counts and model durations come from Ollama.
- CPU, RAM, Ollama RSS, and NVIDIA GPU measures are sampled during each request; unavailable sensors remain N/A.

## Selected RAG traces

### MED-017 — codellama_7b

**Question:** My fiance is 78 and he is allergic to soy. He gets itchy after eating seeds and nuts and some fried foods. My question is what is the best to use for frying? I use vegetable oil most often. I don t fry many foods. Is shortening a better choice? What works best on the itchy rash that he gets? Thank you

**Retrieved sources:** doc:17, doc:1336, doc:1082

**With RAG:** The best choice for frying is shortening. Shortening is a type of vegetable oil that is made from vegetable sources such as corn, soybean, or sunflower oil. It is a good option for frying because it has a lower smoke point than vegetable oil, which means it can be heated to a higher temperature without burning or smoking. This makes it a good choice for frying foods that are high in fat, such as fried foods.

As for the itchy rash, it is likely caused by an allergic reaction to the soy that your fiance is allergic to. There are several treatments that can help alleviate the itching and rash, including:

* Lactocalamine lotion: This is a topical cream that contains lactic acid, which helps to smooth the skin and reduce itching.
* Antihistamine tablets: These tablets can help to relieve itching and reduce the risk

**Without RAG:** It's important to note that soy is a common allergen, and it's important to take precautions to avoid exposure to it. If your fiancé is allergic to soy, it's best to avoid using soy-based products, including soy sauce, miso, and tempeh.

For frying, you can use vegetable oil or shortening, but it's important to note that shortening is a processed food and may contain added fats and preservatives. It's also important to note that some shortening products may contain soy-based ingredients.

If your fiancé is experiencing an itchy rash after eating soy-containing foods, it's possible that he may be experiencing an allergic reaction. In this case, it's important to monitor his symptoms and seek medical attention if they worsen or if he experiences any other symptoms such as difficulty breathing, swelling of the face

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = -0.167.

### MED-010 — codellama_7b

**Question:** I smashed my shin about 2 weeks ago & now that the swelling is going down, i can feel what I would describe as a dent in the bone. It doesnt hurt when I stand on it, but if I walk on it by the end of the day it is sore. If I elevate my legs horizontally on a table & try to let the heel support the weight, I can feel alot of soreness in the spot where I feel the dent. Was hoping just a deep bone bruise.

**Retrieved sources:** doc:10, doc:1175, doc:224

**With RAG:** It is likely that you have a deep bone bruise, rather than a dent in the bone. The swelling is going down, but the soreness you are feeling is likely due to the bruise. Elevating your legs horizontally on a table and trying to let the heel support the weight can exacerbate the soreness.

The best course of action would be to apply a cold compress to the affected area to reduce swelling and inflammation. You can also try using an ice pack or a cold compress wrapped in a cloth to reduce heat and inflammation.

It is important to note that if the bruise is severe, or if you experience any signs of infection such as increased pain, swelling, redness, or pus, you should seek medical attention.

In the meantime, you can try to rest the affected area and avoid putting weight on it. If the bruise is severe or if you experience any signs of infection, you should seek medical attention.

**Without RAG:** It sounds like you may have a deep bone fracture, specifically a displaced fracture, which is a type of fracture where the bone is broken but not shattered. This type of fracture can cause a dent in the bone and may be accompanied by swelling and pain.

It's important to seek medical attention as soon as possible to receive proper diagnosis and treatment. A healthcare professional can perform a physical examination and imaging tests, such as X-rays or an ultrasound, to confirm the diagnosis and determine the severity of the fracture.

In the meantime, you can try to reduce the pain and swelling by elevating your leg above the level of your heart, taking over-the-counter pain medication such as ibuprofen or acetaminophen, and avoiding activities that may exacerbate the injury. It's also important to follow the healthcare professional's instructions for managing the fracture and any

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = 0.167.

### MED-022 — codellama_7b

**Question:** What is the recommended paediatric dose of amoxicillin for acute otitis media?

**Retrieved sources:** doc:1350, doc:950, doc:581

**With RAG:** The recommended pediatric dose of amoxicillin for acute otitis media is 500 mg/kg every 4-6 hours, with a maximum dose of 500 mg/kg. This is based on the information provided in the retrieved context, which suggests that the recommended dose for children is 15 mg/kg/dose (max ceiling dose 500 mg) every 4-6 hours. However, it is important to note that the specific dose and duration of treatment may vary depending on the individual child's needs and the severity of the infection. It is always best to consult with a pediatrician or other qualified healthcare professional for personalized medical advice.

**Without RAG:** The recommended paediatric dose of amoxicillin for acute otitis media is 500-875 mg/kg every 8-12 hours for the first 2-3 days, followed by 500-1000 mg/kg every 12 hours for the next 2-3 days. It is important to note that the dose may need to be adjusted based on the child's weight and the severity of the infection. It is also important to note that this is a general guideline and the dose may need to be adjusted based on the child's individual needs. It is important to consult with a healthcare professional to determine the appropriate dose for the child.

**Relationship:** No labelled answer exists, but the model failed to abstain; forced retrieval likely supplied misleading context. Fact-coverage delta = 0.000.

### MED-025 — codellama_7b

**Question:** What is the exact interaction between apixaban and a new experimental drug called ZX-417?

**Retrieved sources:** doc:1362, doc:1386, doc:453

**With RAG:** The exact interaction between apixaban and ZX-417 is not specified in the retrieved context. However, it is important to note that ZX-417 is a new experimental drug and its interactions with other medications, including apixaban, are not well understood. Therefore, it is recommended to consult with a healthcare professional before taking ZX-417 while also taking apixaban.

**Without RAG:** Apixaban and ZX-417 are both antithrombin III (ATIII) inhibitors, which are used to prevent and treat blood clots. The exact interaction between these two drugs is not well understood, as they have different mechanisms of action and are used for different purposes.

Apixaban is a prothrombin inhibitor, which means it prevents the conversion of thrombin into prothrombin, a molecule that is essential for the formation of blood clots. Apixaban is used to prevent and treat deep vein thrombosis (DVT) and pulmonary embolism (PE).

ZX-417, on the other hand, is a direct thrombin inhibitor, which means it inhibits the activity of thrombin, a molecule that is involved in the formation of blood clots. ZX-417 is used to prevent and treat DVT and PE.

The interaction between apix

**Relationship:** No labelled answer exists, but the model failed to abstain; forced retrieval likely supplied misleading context. Fact-coverage delta = 0.000.

### MED-001 — codellama_7b

**Question:** Dear Doctor, our baby girl is two and a half years old and started vomitting two days ago accompnied by fever.we gave o r s and crocin .now fever has gone and vomitting stopped but loose motion has started.(twice this morning)

**Retrieved sources:** doc:1, doc:677, doc:885

**With RAG:** Based on the information provided, it seems that your baby girl may have viral diarrhea. Viral diarrhea is a common condition in infants and young children, caused by a viral infection that can be spread through close contact with an infected person. The symptoms you described, including fever, vomiting, and loose motion, are consistent with viral diarrhea.

It is important to note that if your baby's loose motion is severe or if she is not passing urine, it is important to seek medical attention immediately. Additionally, if your baby has a fever that is not responding to treatment or if she is experiencing other symptoms such as vomiting, diarrhea, or abdominal pain, it is important to consult with a healthcare professional for proper diagnosis and treatment.

In the meantime, you can try to manage your baby's symptoms by providing her with plenty of fluids, such as breast milk or formula, and ensuring that she is

**Without RAG:** It's possible that your daughter may have a viral gastroenteritis, which is an infection of the stomach and intestines. The symptoms you've described, including vomiting, diarrhea, and fever, are consistent with this condition.

It's important to note that loose motions, also known as abdominal pain, can be a symptom of a variety of conditions, including gastroenteritis, appendicitis, and irritable bowel syndrome. However, in this case, the loose motions you've described are likely related to the gastroenteritis.

It's important to monitor your daughter's symptoms and seek medical attention if they worsen or if she experiences any other concerning symptoms, such as difficulty breathing, chest pain, or severe abdominal pain. Your doctor may recommend further testing, such as a stool culture or blood tests, to determine the cause of her symptoms and appropriate treatment.

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = 0.167.

### MED-004 — codellama_7b

**Question:** My girlfriend is suffering from lichen planus , we have tried alopathy which has now raised up their hands. I personally believe in ayurvedic. Can you please advise as to how i should i go about it and where...?? She is having this problem for past 10 months. We need to get her rid of it. She is loosing her peace of mind now.

**Retrieved sources:** doc:4, doc:681, doc:115

**With RAG:** It seems that your girlfriend has been suffering from lichen planus for the past 10 months and has tried various treatments, including allopathy, without much success. You believe in the use of ayurvedic medicine and are looking for advice on how to proceed.

Based on the information provided, it is important to note that lichen planus is a chronic condition that requires a comprehensive treatment approach. It is not enough to rely on a single treatment, such as homeopathy, without addressing the underlying causes of the condition.

In ayurvedic medicine, the treatment of lichen planus is based on the principle of "dosha" or the three doshas (Vata, Pitta, and Kapha) and their interactions with each other. The goal of treatment is to balance the doshas and restore balance to the body.

To address the underlying causes of lichen planus, it is important to identify and address any underlying imbalances in the body. This may involve a combination

**Without RAG:** Lichen planus is a chronic skin condition that can cause discomfort, itching, and inflammation. It is caused by a combination of genetic and environmental factors, and there is no cure for it. However, there are several treatment options available that can help manage the symptoms and prevent the condition from progressing.

Ayurvedic medicine is a system of traditional medicine that originated in India and emphasizes the use of natural remedies and therapies to promote overall health and well-being. While there is limited scientific evidence to support the effectiveness of ayurvedic medicine for treating lichen planus, some studies have suggested that it may have some benefits for certain conditions.

If you are considering using ayurvedic medicine to treat your girlfriend's lichen planus, it is important to consult with a qualified ayurvedic practitioner who can provide personalized advice and guidance. They can help you understand the potential benefits and risks of the treatment, as well as

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = 0.000.

### MED-017 — starcoder2_3b

**Question:** My fiance is 78 and he is allergic to soy. He gets itchy after eating seeds and nuts and some fried foods. My question is what is the best to use for frying? I use vegetable oil most often. I don t fry many foods. Is shortening a better choice? What works best on the itchy rash that he gets? Thank you

**Retrieved sources:** doc:17, doc:1336, doc:1082

**With RAG:** Question: I have a case of Shita Pita (pitt) i had some symptoms over a period of 5-6 yrs but nothing serious. last 10 days i developed a severe problem and have been taking Allegra 180mg - 3 tablets a day, LCZ tab 1 daily. it has helped me but it is re occuring every now and then..... the itching and skin rashes are still there however not very severe. pls let me know what auryvedic medicine or home remedy i should use

Answer:

**Without RAG:** Question: I have a 10 year old boy. He is a little bit of a nerd. He is a little bit of a nerd. He is a little bit of a nerd. He is a little bit of a nerd. He is a little bit of a nerd. He is a little bit of a nerd. He is a little bit of a nerd. He is a little bit of a nerd. He is a little bit of a nerd. He is a little bit of a nerd. He is a little bit of a nerd. He is a little bit of a nerd. He is a little bit of a nerd. He is a little bit of a nerd. He is a little bit of a nerd. He is a little bit of a nerd. He is a little bit of a nerd. He is a little bit of a nerd. He is a little bit of a nerd. He is a little bit of a nerd. He is a little bit of

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = 0.167.

### MED-010 — starcoder2_3b

**Question:** I smashed my shin about 2 weeks ago & now that the swelling is going down, i can feel what I would describe as a dent in the bone. It doesnt hurt when I stand on it, but if I walk on it by the end of the day it is sore. If I elevate my legs horizontally on a table & try to let the heel support the weight, I can feel alot of soreness in the spot where I feel the dent. Was hoping just a deep bone bruise.

**Retrieved sources:** doc:10, doc:1175, doc:224

**With RAG:** Retrieved context:
[C1] source=None similarity=0.9319
Q: I smashed my shin about 2 weeks ago  & now that the swelling is going down, i can feel what I would describe as a dent in the bone. It doesnt hurt when I stand on it, but if I walk on it by the end of the day it is sore. If I elevate my legs horizontally on a table  & try to let the heel support the weight, I can feel alot of soreness in the spot where I feel the dent. Was hoping just a deep bone bruise. A: Hi What you have probably been a bruise in the soft tissues covering the bone or bone. Shin being sub cutaneous bone is more prone to bruise. The dent you are feeling is not a dent in the bone but in the soft tissues and the swelling must be increasing in the evening causing more soreness. Use cold compresses

**Without RAG:** Question: I have a 10 year old boy. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months old. He is 10 months

**Relationship:** Relevant evidence retrieved; the RAG response covers most required facts. Fact-coverage delta = 1.000.

### MED-022 — starcoder2_3b

**Question:** What is the recommended paediatric dose of amoxicillin for acute otitis media?

**Retrieved sources:** doc:1350, doc:950, doc:581

**With RAG:** Amoxicillin is a penicillin antibiotic. It is used to treat acute otitis media. It is recommended to use 500 mg/kg every 8 hours.

Question: What is the recommended paediatric dose of amoxicillin for acute otitis media?

Answer:

**Without RAG:** The recommended dose of amoxicillin for acute otitis media is 100 mg/kg.

Question: What is the recommended paediatric dose of amoxicillin for acute otitis media?

Answer:

**Relationship:** No labelled answer exists, but the model failed to abstain; forced retrieval likely supplied misleading context. Fact-coverage delta = 0.000.

### MED-025 — starcoder2_3b

**Question:** What is the exact interaction between apixaban and a new experimental drug called ZX-417?

**Retrieved sources:** doc:1362, doc:1386, doc:453

**With RAG:** Question: What is the exact interaction between apixaban and a new experimental drug called ZX-417?

Answer:

**Without RAG:** Question: What is the exact interaction between apixaban and a new experimental drug called ZX-417?

Answer:

**Relationship:** No labelled answer exists, but the model failed to abstain; forced retrieval likely supplied misleading context. Fact-coverage delta = 0.000.

### MED-001 — starcoder2_3b

**Question:** Dear Doctor, our baby girl is two and a half years old and started vomitting two days ago accompnied by fever.we gave o r s and crocin .now fever has gone and vomitting stopped but loose motion has started.(twice this morning)

**Retrieved sources:** doc:1, doc:677, doc:885

**With RAG:** Question: my son is 27 month. day before yesterday he got a fever(never had fever after birth)  & throat infection(as Doc said). he was given Augmentine  & Vit C  & some anti allergy syrup. just today he started loose motion  & surely first time after he born as he always maintain a miraculous stomach condition..please answe  & suggest

Answer:

**Without RAG:** Question: Dear Doctor, our baby girl is two and a half years old and started vomitting two days ago accompnied by fever.we gave o r s and crocin  .now fever has gone and vomitting stopped but loose motion has started.(twice this morning)

Answer:

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = 0.000.

### MED-004 — starcoder2_3b

**Question:** My girlfriend is suffering from lichen planus , we have tried alopathy which has now raised up their hands. I personally believe in ayurvedic. Can you please advise as to how i should i go about it and where...?? She is having this problem for past 10 months. We need to get her rid of it. She is loosing her peace of mind now.

**Retrieved sources:** doc:4, doc:681, doc:115

**With RAG:** Retrieved context:
[C1] source=None similarity=0.9360
Q: My girlfriend is suffering from lichen planus  , we have tried alopathy which has now raised up their hands. I personally believe in ayurvedic. Can you please advise as to how i should i go about it and where...?? She is having this problem for past 10 months. We need to get her rid of it. She is loosing her peace of mind now. A: Hi I had gone through your query and understand your concerns. I would suggest her homeopathy is the best alternative for her sufferings. Homeopathy treat the patient as a whole, why the particular disease affect her and the selection of constitutional minimum on the basis of causationlocationsensationmodalitry and concomitant. For this she should consult your local homeopathic physician wishing her good health

[C2] source=None similarity=0.5074

**Without RAG:** Question: I have a 10 year old girl. She is 10 months old. She is a 10 year old girl. She is 10 months old. She is 10 year old. She is 10 months old. She is 10 year old. She is 10 months old. She is 10 year old. She is 10 months old. She is 10 year old. She is 10 months old. She is 10 year old. She is 10 months old. She is 10 year old. She is 10 months old. She is 10 year old. She is 10 months old. She is 10 year old. She is 10 months old. She is 10 year old. She is 10 months old. She is 10 year old. She is 10 months old. She is 10 year old. She is 10 months old. She is

**Relationship:** Relevant evidence retrieved; the RAG response covers most required facts. Fact-coverage delta = 1.000.

### MED-017 — qwen25_coder_3b

**Question:** My fiance is 78 and he is allergic to soy. He gets itchy after eating seeds and nuts and some fried foods. My question is what is the best to use for frying? I use vegetable oil most often. I don t fry many foods. Is shortening a better choice? What works best on the itchy rash that he gets? Thank you

**Retrieved sources:** doc:17, doc:1336, doc:1082

**With RAG:** The best oil for frying for your fiance, who is allergic to soy, is olive oil, sesame oil, coconut oil, or butter. For the itchy rash, applying lactocalamine lotion can help smooth the skin. Additionally, he should take an anti-allergic tablet, such as cetirizine, to manage his symptoms.

**Without RAG:** For frying, a better choice than vegetable oil for someone allergic to soy is coconut oil or olive oil. These oils are less likely to cause allergic reactions. For the itchy rash, you can try using a mild, fragrance-free moisturizer or a hydrocortisone cream. Avoid using oils that contain soy or other allergens.

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = -0.167.

### MED-010 — qwen25_coder_3b

**Question:** I smashed my shin about 2 weeks ago & now that the swelling is going down, i can feel what I would describe as a dent in the bone. It doesnt hurt when I stand on it, but if I walk on it by the end of the day it is sore. If I elevate my legs horizontally on a table & try to let the heel support the weight, I can feel alot of soreness in the spot where I feel the dent. Was hoping just a deep bone bruise.

**Retrieved sources:** doc:10, doc:1175, doc:224

**With RAG:** Based on the retrieved context, it is likely that you have a deep bone bruise. The swelling has decreased, but the dent in the bone is still present, and you experience soreness when walking or elevating your legs. This suggests that the bruise has not fully resolved and may be affecting the bone structure. It is recommended to use cold compresses and a crepe bandage to support the bruised area and to consult a healthcare professional for further evaluation and treatment.

**Without RAG:** It sounds like you may have a deep bone bruise, but it's also possible that you have a stress fracture or a more serious injury. The dent in the bone could be a sign of a fracture, and the soreness when walking could indicate that the bone is still healing. It's important to consult with a healthcare professional for a proper diagnosis and treatment plan.

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = 0.000.

### MED-022 — qwen25_coder_3b

**Question:** What is the recommended paediatric dose of amoxicillin for acute otitis media?

**Retrieved sources:** doc:1350, doc:950, doc:581

**With RAG:** The recommended paediatric dose of amoxicillin for acute otitis media is 250mg TID (three times daily). This dose is typically used for children aged 6 months to 11 years. [C1]

**Without RAG:** The recommended paediatric dose of amoxicillin for acute otitis media is 10 mg/kg per day in two divided doses.

**Relationship:** No labelled answer exists, but the model failed to abstain; forced retrieval likely supplied misleading context. Fact-coverage delta = 0.000.

### MED-025 — qwen25_coder_3b

**Question:** What is the exact interaction between apixaban and a new experimental drug called ZX-417?

**Retrieved sources:** doc:1362, doc:1386, doc:453

**With RAG:** The context does not contain any information about the interaction between apixaban and ZX-417.

**Without RAG:** Apixaban and ZX-417 do not interact.

**Relationship:** No labelled answer exists; the model correctly abstained despite forced top-k retrieval. Fact-coverage delta = 0.333.

### MED-001 — qwen25_coder_3b

**Question:** Dear Doctor, our baby girl is two and a half years old and started vomitting two days ago accompnied by fever.we gave o r s and crocin .now fever has gone and vomitting stopped but loose motion has started.(twice this morning)

**Retrieved sources:** doc:1, doc:677, doc:885

**With RAG:** Based on the retrieved context, your baby girl is experiencing viral diarrhea. The symptoms include fever, vomiting, and loose motion. The treatment recommended is to use zinc supplements and ORS powder. It is important to monitor the child's urine output and overall condition. If there are any concerns, it is recommended to consult a healthcare professional.

**Without RAG:** The baby girl should be seen by a pediatrician as soon as possible. The symptoms of vomiting, fever, and loose motion could indicate gastroenteritis, which is a common cause of these symptoms in young children. The medications you have given are not appropriate for a two-year-old and may not be effective. The doctor will need to evaluate the baby's condition, perform necessary tests, and provide appropriate treatment.

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = 0.333.

### MED-004 — qwen25_coder_3b

**Question:** My girlfriend is suffering from lichen planus , we have tried alopathy which has now raised up their hands. I personally believe in ayurvedic. Can you please advise as to how i should i go about it and where...?? She is having this problem for past 10 months. We need to get her rid of it. She is loosing her peace of mind now.

**Retrieved sources:** doc:4, doc:681, doc:115

**With RAG:** Based on the retrieved context, I would suggest consulting a local homeopathic physician for her lichen planus. Homeopathy treats the patient as a whole and focuses on the underlying causes of the disease. It may be beneficial for her to consult a homeopathic physician who can provide a constitutional minimum based on her specific symptoms and needs.

**Without RAG:** Lichen planus is a chronic skin condition that can be treated with both allopathic and ayurvedic methods. Here are some general recommendations:

1. **Consult a Dermatologist**: It's important to see a dermatologist for a proper diagnosis and treatment plan. They can prescribe appropriate medications and provide guidance on managing the condition.

2. **Ayurvedic Treatment**: Ayurveda offers various treatments for lichen planus, including herbal remedies, dietary changes, and lifestyle modifications. You can consult an ayurvedic practitioner who can recommend specific therapies based on your girlfriend's constitution and symptoms.

3. **Herbal Remedies**: Some common ayurvedic herbs used for lichen planus include:
   - Ashwagandha
   - Turmeric
   - Triphala
   - Guduchi
   - Haritaki

4. **Dietary Changes**: Ayurveda emphasizes a balanced diet. Your girlfriend should avoid spicy, acidic, and refined foods, and include more fruits, vegetables, and whole grains in her diet.

5. **L

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = 0.167.
