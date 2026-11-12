# Task 4 — A better leaf classifier

## Goal and approach

The goal was to obtain a classifier of four leaf species (apple, cherry, chestnut, maple) that works as well as possible on photos it did not see during training. I therefore made all decisions only on the validation set, and I used the test set only once, at the very end, for the baseline and for the model chosen in advance.

### Data

I combined the photos of my group (`data/leaves`, 306 photos) and the photos from previous years (`data/old_leaves`, 820 photos). All of them are 96 × 96 px.

Before training, I cleaned the data and removed 22 images:

- **21 exact duplicates.** All of them were in the old data. Four images were in both the old training and the old validation folder, and 13 duplicates were within the old training folder. Two pairs were the same image with two different classes (apple and cherry), so I removed both copies.
- **1 photo of a twig** with several leaves at once (`leaves/val/apple/apple_007`).

I checked the labels with cross-prediction: a model trained on one half of the data predicted the other half and vice versa. Of 1017 images, 105 were misclassified with a confidence above 0.9, almost all of them as an apple ↔ cherry confusion. After reviewing them, I did not change the labels. These are not wrongly labelled photos but a pair of species that are hard to tell apart at 96 px.

### Data split

The split is made by the script `make_splits.py` (seed 0):

| set | my group | old data | total | use |
| --- | --- | --- | --- | --- |
| train | 244 | 683 | 927 | training |
| val | 32 | 58 | 90 | choosing the variants and the final model |
| test | 29 | 58 | 87 | a single evaluation at the end |

The training set consists of both training folders. The validation and test sets are two random halves of the teacher's validation folders, stratified by source and class.

Besides the ordinary validation, I also evaluated every variant on a **camera-like validation**. It was created from the validation images, which I randomly cropped (RandomResizedCrop 60–100 %), rotated (±20°) and changed in colour (ColorJitter 0.3 / 0.3 / 0.3 / 0.05) once with a fixed seed. Every image gave 3 versions, 270 images in total. This set imitates frames from a phone camera.

### Fair comparison

First I trained the baseline B0, then variants that each change only one thing compared with B0, and finally two combinations. I trained every variant with seeds 0, 1 and 2 on the same training images. In the tables I give the mean ± standard deviation over these seeds.

Before running the combinations, I set a selection rule: the final model will be the combination with the highest mean of (val + camera-like) / 2. This prevented me from choosing the model according to what I liked after seeing the results.

**Size of the noise:** one image represents 1.1 % of the validation set, 1.1 % of the test set and 0.4 % of the camera-like validation. The camera-like set, however, consists of only three versions of the same 90 leaves, so its real precision is lower than the number of images suggests. The standard deviation over seeds is 4–7 points for most variants, which corresponds to 4–6 validation images. I therefore consider a difference smaller than about one standard deviation to be noise.

## Results on validation

**Baseline B0:** pretrained `mobilenet_v3_small` (ImageNet), the last layer replaced by a layer with 4 outputs, the whole network fine-tuned, Adam with lr 1e-3, 15 epochs, batch 32, no augmentation.

| variant | change compared with B0 | val | camera-like |
| --- | --- | --- | --- |
| B0 | — | 0.863 ± 0.046 | 0.800 ± 0.042 |
| A1 | geometric augmentation + ColorJitter | 0.863 ± 0.065 | 0.872 ± 0.050 |
| A2 | cosine learning-rate schedule, 30 epochs | 0.907 ± 0.006 | 0.815 ± 0.007 |
| A3 | input enlarged to 224 px inside the model | 0.893 ± 0.074 | 0.784 ± 0.030 |
| A4 | `mobilenet_v3_large` | 0.896 ± 0.057 | 0.741 ± 0.110 |
| A5 | `efficientnet_b0` | 0.907 ± 0.026 | 0.858 ± 0.011 |
| A6 | `resnet18` | 0.878 ± 0.059 | 0.714 ± 0.063 |
| A7 | lr 1e-4 | 0.900 ± 0.044 | 0.795 ± 0.008 |
| C1 | A5 + A1 + A2 | **0.952 ± 0.017** | **0.949 ± 0.004** |
| C2 | A5 + A2 | 0.926 ± 0.006 | 0.846 ± 0.019 |

Score according to the selection rule, (val + camera-like) / 2: B0 0.831 ± 0.027, A5 0.883 ± 0.018, **C1 0.951 ± 0.008**, C2 0.886 ± 0.012. The final model is therefore C1.

Size of the models in ONNX:

| network | parameters | ONNX |
| --- | --- | --- |
| `mobilenet_v3_small` (B0) | 1.5 M | 6.4 MB |
| `mobilenet_v3_large` (A4) | 4.2 M | 17.2 MB |
| `efficientnet_b0` (A5) | 4.0 M | 16.7 MB |
| `resnet18` (A6) | 11.2 M | 44.8 MB |

A3 has the same number of parameters as B0, but about 5.4 times more computation.

## What the individual changes brought

**Augmentation (A1)** brought nothing on the ordinary validation: 0.863, the same as B0. On the camera-like validation, it increased the accuracy by 7 points, which is about 19 of 270 images. This gain must be taken with caution, because the camera-like set was created with the same type of transformations that the augmentation uses during training. Part of the improvement is therefore given by the construction of the metric itself.

**The cosine learning-rate schedule with 30 epochs (A2)** improved the validation by 4.4 points (about 4 images) and had the smallest spread of all variants (± 0.006). On the camera-like validation it brought only a small difference within the noise. Training with a gradually decreasing learning rate therefore stabilizes the result, but does not increase the robustness to changes of the frame.

**Choice of the network (A4–A6):** only `efficientnet_b0` (A5) was better both on the validation (+4.4 points) and on the camera-like validation (+5.8 points), and with a small spread. A bigger network does not help by itself: `mobilenet_v3_large` and `resnet18` reached worse and less stable results on the camera-like validation, and `resnet18` moreover has 45 MB, which is unnecessarily large for a phone.

**The combination C1** (efficientnet_b0, augmentation, cosine schedule, 30 epochs) was clearly better than each change on its own: val 0.952 compared with 0.907 for A5 alone and 0.926 for C2 without augmentation. The comparison of C1 and C2 shows that with EfficientNet, augmentation helped also on the ordinary validation (+2.6 points), not only on the camera-like validation, where it is favoured. Augmentation makes the task harder, so the network needs longer training, and that is why both changes work best together. C1 also had the smallest spread on the camera-like validation (± 0.004).

## What did not help

- **Input enlarged to 224 px (A3):** validation within the noise (spread ± 0.074), camera-like validation 1.6 points worse and about 5.4 times more computation. The photos have only 96 px, so enlarging them adds no new details.
- **`mobilenet_v3_large` (A4):** camera-like validation 5.9 points worse, with a large spread (± 0.110).
- **`resnet18` (A6):** camera-like validation 8.6 points worse and the model 7 times larger than B0.
- **A lower learning rate of 1e-4 (A7):** the difference on both validations is within the noise.

## Final test

On the test set (87 images) I evaluated only B0 and the C1 chosen in advance, and only once. After the evaluation I did not change or retrain anything.

| model | seed 0 | seed 1 | seed 2 | mean ± std | my group (29) | old data (58) |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | 0.862 | 0.828 | 0.897 | 0.862 ± 0.034 | 0.851 ± 0.053 | 0.868 ± 0.050 |
| C1 | 0.931 | 0.920 | 0.897 | **0.916 ± 0.018** | 0.954 ± 0.020 | 0.897 ± 0.017 |

Confusion matrices for seed 0 (rows = true class, columns = prediction):

| C1 | apple | cherry | chestnut | maple |
| --- | --- | --- | --- | --- |
| apple | 22 | 1 | 0 | 0 |
| cherry | 5 | 19 | 0 | 0 |
| chestnut | 0 | 0 | 20 | 0 |
| maple | 0 | 0 | 0 | 20 |

| B0 | apple | cherry | chestnut | maple |
| --- | --- | --- | --- | --- |
| apple | 17 | 6 | 0 | 0 |
| cherry | 4 | 20 | 0 | 0 |
| chestnut | 0 | 1 | 19 | 0 |
| maple | 0 | 0 | 1 | 19 |

**On the test set, C1 is better than B0 by 5.4 points**, which corresponds to about 4–5 of 87 images. For two seeds C1 is better and for the third both models are equal (0.897). C1 also has half the spread (± 0.018 compared with ± 0.034), so it is more stable. With three seeds this is not a statistical proof, but the improvement has a consistent direction and agrees with the validation.

**The difference between validation and test for C1 is only 3.6 points** (0.952 → 0.916), i.e. about 3 images. The validation therefore estimated the test performance well, and the model selection was not noticeably overfitted to the validation set.

**The remaining errors of C1 are only apple ↔ cherry confusions** (6 errors, mostly cherry labelled as apple). It classifies chestnut and maple without errors. The leaves of both apple and cherry are ovate with a serrated margin. The features that distinguish them (the glands on the cherry's petiole, its doubly serrated margin and elongated tip) are mostly lost at a resolution of 96 × 96 px. B0 made the same type of errors, only more often (10 apple ↔ cherry confusions).

### Limitations

- The test photos come from the same two sources as the training photos. The training set may therefore contain other photos of the same trees, phones and photo sessions. The test is therefore more optimistic than it would be with photos from a completely new person with a different phone.
- The chestnut class contains mainly horse chestnut (palmately compound leaves) and a few single serrated leaves, so it is not a completely uniform class.

## Model for the phone

I would put **C1** on the phone: `efficientnet_b0` with augmentation (random crop, flips, rotation, ColorJitter), a cosine learning-rate schedule and 30 epochs. I exported the version with seed 0, which I chose in advance and not according to the test result, as `leaf_model_final.onnx` (16.7 MB, input 1 × 3 × 96 × 96). It is the best model on the validation, on the camera-like validation and on the test set, and at the same time the most stable across seeds. At 16.7 MB it is still small enough to be loaded in a phone browser. On my page it is the default, and the switch lets you go back to the Task 2 model.

**Test on the phone:** On my phone I tried all leaf classes, and the Task 4 model works much better than the Task 2 model. For chestnut and maple it works flawlessly, and for apple and cherry it is in some cases not 100 % sure, but it still classifies well. As written above, on the phone the key is to bring the leaf close in the frame to improve the classification, ideally with no other leaves in the background.

---

*This text was written and checked by me and then translated into English with the help of AI. My original Slovak version follows below.*

*Tento text som napísal a skontroloval ja a následne bol pomocou AI preložený do angličtiny. Nižšie nasleduje moja pôvodná slovenská verzia.*

---

# Task 4 — Lepší klasifikátor listov

## Cieľ a postup

Cieľom bolo získať klasifikátor štyroch druhov listov (jabloň, čerešňa, gaštan, javor), ktorý funguje čo najlepšie na fotkách, ktoré pri tréningu nevidel. Všetky rozhodnutia som preto robil len na validačnej množine a testovaciu množinu som použil iba raz, na úplnom konci, pre baseline a vopred vybraný model.

### Dáta

Spojil som fotky mojej skupiny (`data/leaves`, 306 fotiek) a fotky z minulých rokov (`data/old_leaves`, 820 fotiek). Všetky majú rozmer 96 × 96 px.

Pred tréningom som dáta vyčistil a vyradil 22 obrázkov:

- **21 presných duplikátov**. Všetky boli v starých dátach. Štyri obrázky boli súčasne v starom trénovacom aj validačnom priečinku a 13 duplikátov bolo v rámci starého trénovacieho priečinka. Dve dvojice tvoril ten istý obrázok s dvoma rôznymi triedami (apple a cherry), preto som vyradil obe kópie.
- **1 fotku s vetvičkou**, na ktorej je viac listov naraz (`leaves/val/apple/apple_007`).

Štítky som skontroloval krížovou predikciou: model trénovaný na jednej polovici dát predikoval druhú polovicu a naopak. Zo 1017 obrázkov bolo 105 klasifikovaných chybne s istotou nad 0,9, takmer všetky ako zámena jablko ↔ čerešňa. Po prezretí som štítky nemenil. Nejde o chybne zaradené fotky, ale o dvojicu druhov, ktoré sa pri 96 px ťažko rozlišujú.

### Rozdelenie dát

Rozdelenie robí skript `make_splits.py` (seed 0):

| množina | moja skupina | staré dáta | spolu | použitie |
| --- | --- | --- | --- | --- |
| train | 244 | 683 | 927 | tréning |
| val | 32 | 58 | 90 | výber variantov a finálneho modelu |
| test | 29 | 58 | 87 | jediné vyhodnotenie na konci |

Trénovaciu množinu tvoria oba trénovacie priečinky. Validačná a testovacia množina sú dve náhodné polovice učiteľových validačných priečinkov, stratifikované podľa zdroja a triedy.

Okrem bežnej validácie som každý variant vyhodnotil aj na **camera-like validácii**. Tá vznikla z validačných obrázkov, ktoré som raz s pevným seedom náhodne orezal (RandomResizedCrop 60–100 %), otočil (±20°) a farebne zmenil (ColorJitter 0,3 / 0,3 / 0,3 / 0,05). Z každého obrázka vznikli 3 verzie, spolu 270 obrázkov. Táto množina napodobňuje zábery z kamery mobilu.

### Férové porovnanie

Najprv som natrénoval baseline B0, potom varianty, z ktorých každý mení oproti B0 iba jednu vec, a nakoniec dve kombinácie. Každý variant som trénoval so seedmi 0, 1 a 2 na rovnakých trénovacích obrázkoch. V tabuľkách uvádzam priemer ± smerodajnú odchýlku cez tieto seedy.

Pred spustením kombinácií som si stanovil pravidlo výberu: finálny model bude kombinácia s najvyšším priemerom (val + camera-like) / 2. Takto som zabránil tomu, aby som model vyberal podľa toho, čo sa mi po výsledkoch páči.

**Veľkosť šumu:** jeden obrázok predstavuje 1,1 % validácie, 1,1 % testu a 0,4 % camera-like validácie. Camera-like sú však len tri verzie tých istých 90 listov, preto je jej skutočná presnosť menšia, ako naznačuje počet obrázkov. Smerodajná odchýlka cez seedy je pri väčšine variantov 4–7 bodov, čo zodpovedá 4–6 validačným obrázkom. Rozdiel menší ako približne jedna smerodajná odchýlka preto považujem za šum.

## Výsledky na validácii

**Baseline B0:** predtrénovaná `mobilenet_v3_small` (ImageNet), posledná vrstva nahradená vrstvou so 4 výstupmi, dotrénovaná celá sieť, Adam s lr 1e-3, 15 epoch, batch 32, bez augmentácie.

| variant | zmena oproti B0 | val | camera-like |
| --- | --- | --- | --- |
| B0 | — | 0,863 ± 0,046 | 0,800 ± 0,042 |
| A1 | geometrická augmentácia + ColorJitter | 0,863 ± 0,065 | 0,872 ± 0,050 |
| A2 | kosínusový plán učenia, 30 epoch | 0,907 ± 0,006 | 0,815 ± 0,007 |
| A3 | zväčšenie vstupu na 224 px vnútri modelu | 0,893 ± 0,074 | 0,784 ± 0,030 |
| A4 | `mobilenet_v3_large` | 0,896 ± 0,057 | 0,741 ± 0,110 |
| A5 | `efficientnet_b0` | 0,907 ± 0,026 | 0,858 ± 0,011 |
| A6 | `resnet18` | 0,878 ± 0,059 | 0,714 ± 0,063 |
| A7 | lr 1e-4 | 0,900 ± 0,044 | 0,795 ± 0,008 |
| C1 | A5 + A1 + A2 | **0,952 ± 0,017** | **0,949 ± 0,004** |
| C2 | A5 + A2 | 0,926 ± 0,006 | 0,846 ± 0,019 |

Skóre podľa pravidla výberu, (val + camera-like) / 2: B0 0,831 ± 0,027, A5 0,883 ± 0,018, **C1 0,951 ± 0,008**, C2 0,886 ± 0,012. Finálnym modelom sa teda stal C1.

Veľkosť modelov v ONNX:

| sieť | parametre | ONNX |
| --- | --- | --- |
| `mobilenet_v3_small` (B0) | 1,5 M | 6,4 MB |
| `mobilenet_v3_large` (A4) | 4,2 M | 17,2 MB |
| `efficientnet_b0` (A5) | 4,0 M | 16,7 MB |
| `resnet18` (A6) | 11,2 M | 44,8 MB |

A3 má rovnaký počet parametrov ako B0, ale približne 5,4-krát viac výpočtov.

## Čo priniesli jednotlivé zmeny

**Augmentácia (A1)** na bežnej validácii nepriniesla nič: 0,863, rovnako ako B0. Na camera-like validácii zvýšila presnosť o 7 bodov, čo je približne 19 z 270 obrázkov. Tento zisk treba brať opatrne, pretože camera-like množina vznikla z rovnakého typu transformácií, aké augmentácia používa pri tréningu. Časť zlepšenia je preto daná samotnou konštrukciou metriky.

**Kosínusový plán učenia s 30 epochami (A2)** zlepšil validáciu o 4,4 bodu (približne 4 obrázky) a mal zo všetkých variantov najmenší rozptyl (± 0,006). Na camera-like validácii priniesol iba malý rozdiel v rámci šumu. Učenie s postupne klesajúcim learning rate teda stabilizuje výsledok, ale robustnosť voči zmenám záberu nezvyšuje.

**Výber siete (A4–A6):** jedine `efficientnet_b0` (A5) bola lepšia na validácii (+4,4 bodu) aj na camera-like validácii (+5,8 bodu), a to s malým rozptylom. Väčšia sieť sama osebe nepomáha: `mobilenet_v3_large` aj `resnet18` dosiahli na camera-like validácii horšie a menej stabilné výsledky a `resnet18` má navyše 45 MB, čo je pre mobil zbytočne veľa.

**Kombinácia C1** (efficientnet_b0, augmentácia, kosínusový plán, 30 epoch) bola výrazne lepšia ako každá zmena samostatne: val 0,952 oproti 0,907 pre samotný A5 a 0,926 pre C2 bez augmentácie. Porovnanie C1 a C2 ukazuje, že pri efficientnete augmentácia pomohla aj na bežnej validácii (+2,6 bodu), nielen na camera-like validácii, kde je zvýhodnená. Augmentácia sťažuje úlohu, takže sieť potrebuje dlhší tréning, a preto obe zmeny fungujú najlepšie spolu. C1 mal zároveň najmenší rozptyl na camera-like validácii (± 0,004).

## Čo nepomohlo

- **Zväčšenie vstupu na 224 px (A3):** validácia v rámci šumu (rozptyl ± 0,074), camera-like validácia o 1,6 bodu horšia a približne 5,4-krát viac výpočtov. Fotky majú len 96 px, takže zväčšenie nepridáva žiadne nové detaily.
- **`mobilenet_v3_large` (A4):** camera-like validácia o 5,9 bodu horšia, s veľkým rozptylom (± 0,110).
- **`resnet18` (A6):** camera-like validácia o 8,6 bodu horšia a model 7-krát väčší ako B0.
- **Nižší learning rate 1e-4 (A7):** rozdiel na oboch validáciách je v rámci šumu.

## Finálny test

Na testovacej množine (87 obrázkov) som vyhodnotil iba B0 a vopred vybraný C1, a to raz. Po vyhodnotení som už nič nemenil ani nepretrénoval.

| model | seed 0 | seed 1 | seed 2 | priemer ± sm. odch. | moja skupina (29) | staré dáta (58) |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | 0,862 | 0,828 | 0,897 | 0,862 ± 0,034 | 0,851 ± 0,053 | 0,868 ± 0,050 |
| C1 | 0,931 | 0,920 | 0,897 | **0,916 ± 0,018** | 0,954 ± 0,020 | 0,897 ± 0,017 |

Matice zámen pre seed 0 (riadky = skutočná trieda, stĺpce = predikcia):

| C1 | apple | cherry | chestnut | maple |
| --- | --- | --- | --- | --- |
| apple | 22 | 1 | 0 | 0 |
| cherry | 5 | 19 | 0 | 0 |
| chestnut | 0 | 0 | 20 | 0 |
| maple | 0 | 0 | 0 | 20 |

| B0 | apple | cherry | chestnut | maple |
| --- | --- | --- | --- | --- |
| apple | 17 | 6 | 0 | 0 |
| cherry | 4 | 20 | 0 | 0 |
| chestnut | 0 | 1 | 19 | 0 |
| maple | 0 | 0 | 1 | 19 |

**C1 je na teste lepší ako B0 o 5,4 bodu**, čo zodpovedá približne 4–5 obrázkom z 87. Pri dvoch seedoch je C1 lepší a pri treťom sú oba modely rovnaké (0,897). C1 má navyše polovičný rozptyl (± 0,018 oproti ± 0,034), takže je stabilnejší. Pri troch seedoch nejde o štatistický dôkaz, no zlepšenie má konzistentný smer a zhoduje sa s validáciou.

**Rozdiel medzi validáciou a testom je pri C1 len 3,6 bodu** (0,952 → 0,916), teda približne 3 obrázky. Validácia teda odhadla výkon na teste dobre a výber modelu nebol výrazne preučený na validačnú množinu.

**Zostávajúce chyby C1 sú iba zámeny jablko ↔ čerešňa** (6 chýb, prevažne čerešňa označená ako jablko). Gaštan a javor klasifikuje bez chyby. Listy jablone aj čerešne sú vajcovité s pílkovitým okrajom. Znaky, ktoré ich odlišujú (žliazky na stopke čerešne, jej dvojito pílkovitý okraj a predĺžená špička), sa pri rozlíšení 96 × 96 px zväčša strácajú. B0 robil rovnaký typ chýb, len častejšie (10 zámen jablko ↔ čerešňa).

### Obmedzenia

- Testovacie fotky pochádzajú z rovnakých dvoch zdrojov ako trénovacie. V trénovacej množine teda môžu byť iné fotky z tých istých stromov, mobilov a fotení. Test je preto optimistickejší, ako by bol pri fotkách od úplne nového človeka s iným mobilom.
- Trieda chestnut obsahuje najmä pagaštan (dlaňovito zložené listy) a niekoľko jednotlivých pílkovitých listov, takže nejde o úplne jednotnú triedu.

## Model na mobil

Na mobil by som nasadil **C1**: `efficientnet_b0` s augmentáciou (náhodný výrez, prevrátenie, rotácia, ColorJitter), kosínusovým plánom učenia a 30 epochami. Exportoval som verziu so seedom 0, ktorý som zvolil vopred, nie podľa výsledku testu, ako `leaf_model_final.onnx` (16,7 MB, vstup 1 × 3 × 96 × 96). Je to najlepší model na validácii, na camera-like validácii aj na teste a zároveň najstabilnejší cez seedy. Pri 16,7 MB je stále dostatočne malý na načítanie v prehliadači mobilu. Na mojej stránke je predvolený a prepínačom sa dá vrátiť na model z Tasku 2.

**Skúška na mobile:** Na mobile som skúšal všetky triedy listov a model z Tasku 4 funguje omnoho lepšie ako model z Tasku 2. Pri chestnut a maple funguje bezchybne a pri apple a cherry si v niektorých prípadoch nie je na 100 % istý, ale aj napriek tomu klasifikuje kvalitne. Ako je vyššie napísané, na mobile je základ priblížiť list do záberu pre zlepšenie klasifikácie, a ideálne, ak v pozadí nie sú iné listy.
