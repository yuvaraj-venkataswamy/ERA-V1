# ERA V1 - Session 16 Assignment

## Folder Structure
```
session5
|── model.py
|── train.py
|── S16.ipynb     
|── README.md   
```

## One Cycle Policy (OCP)
OCP Sets the learning rate of each parameter group according to the 1cycle learning rate policy. The 1 cycle policy anneals the learning rate from an initial learning rate to some maximum learning rate and then from that maximum learning rate to some minimum learning rate much lower than the initial learning rate.

## Automatic Mixed Precision (AMP)
Numeric Precision Reduction means yielding speedups through the use of floating point reduction and quantization. This is perhaps the most general method for yielding prediction-time speedups.

## Parameter Sharing (PS)
Parameter sharing method is faster than using the same parameters for all layers such as Universal Transformers. Universal Transformers raise their expressiveness power by increasing the size of weight matrices for each layer.

## Training
```
Epoch 00: 100%|██████████| 966/966 [03:06<00:00,  5.19it/s, loss=5.101]
Average loss of epoch 0 is 5.788023749246854
--------------------------------------------------------------------------------
    SOURCE: These words pronounced, the chief officer went below again.
    TARGET: Ces mots prononcés, le second redescendit.
 PREDICTED: Ces mots , les voix , en face .
--------------------------------------------------------------------------------
    SOURCE: Hans and my uncle, clinging to the wall, tried to nibble a few bits of biscuit.
    TARGET: Hans et mon oncle, accotés à la paroi, essayèrent de grignoter quelques morceaux de biscuit. De longs gémissements s'échappaient de mes lèvres tuméfiées.
 PREDICTED: Ned Land et me leva , en face , en face de la coque de granit .
--------------------------------------------------------------------------------
Epoch 01: 100%|██████████| 966/966 [03:03<00:00,  5.27it/s, loss=3.847]
Average loss of epoch 1 is 4.320783610669722
--------------------------------------------------------------------------------
    SOURCE: "Well, admitting all your calculations to be quite correct, you must allow me to draw one rigid result therefrom."
    TARGET: «Mon oncle, repris-je, je tiens pour exact tous vos calculs, mais permettez-moi d'en tirer une conséquence rigoureuse.
 PREDICTED: -- Eh bien , tout cela vous a fait de votre situation , vous me devez être un seul de l ' état .
--------------------------------------------------------------------------------
    SOURCE: I felt that.
    TARGET: Je le sentis.
 PREDICTED: Je sentis que je sentis .
--------------------------------------------------------------------------------
Epoch 02: 100%|██████████| 966/966 [03:04<00:00,  5.23it/s, loss=3.700]
Average loss of epoch 2 is 3.662091069833586
--------------------------------------------------------------------------------
    SOURCE: "We have all the necessary materials for making a battery, and the most difficult thing will be to stretch the wires, but by means of a drawplate I think we shall manage it."
    TARGET: Nous avons tous les éléments nécessaires pour confectionner une pile, et le plus difficile sera d'étirer des fils de fer, mais au moyen d'une filière, je pense que nous en viendrons à bout.
 PREDICTED: Nous avons tout à tout le monde pour faire une façon de plus et de plus difficile à l ' entendre , mais je le par un .
--------------------------------------------------------------------------------
    SOURCE: He went back to the cab, which the cabman drew up again, and he pulled out a little black wooden box, which he carried off under his arm.
    TARGET: Il revint a la voiture, que le cocher remisait, et il tira du coffre une petite caisse de bois noir, qu'il emporta sous son bras.
 PREDICTED: Il alla chercher le fiacre , qui se mit à se retirer , et il tira un petit petit bras noir qui tenait sous le bras .
--------------------------------------------------------------------------------
Epoch 03: 100%|██████████| 966/966 [03:06<00:00,  5.18it/s, loss=3.450]
Average loss of epoch 3 is 3.493731292631809
--------------------------------------------------------------------------------
    SOURCE: The commander of the fort was anxious, though he tried to conceal his apprehensions.
    TARGET: Le capitaine du fort Kearney était très inquiet, bien qu'il ne voulût rien laisser paraître de son inquiétude.
 PREDICTED: Le commandant de la pensée était fort , mais il voulut retenir son esprit .
--------------------------------------------------------------------------------
    SOURCE: The archdeacon had just seated himself, by the light of a three−jetted copper lamp, before a vast coffer crammed with manuscripts.
    TARGET: L’archidiacre venait de s’asseoir à la clarté d’un trois-becs de cuivre devant un vaste bahut chargé de manuscrits.
 PREDICTED: L ’ archidiacre avait été assis , par la lumière de trois de la lampe , devant une vaste immense immense salle à vapeur .
--------------------------------------------------------------------------------
Epoch 04: 100%|██████████| 966/966 [03:04<00:00,  5.24it/s, loss=3.525]
Average loss of epoch 4 is 3.3623336560977912
--------------------------------------------------------------------------------
    SOURCE: There was no reply to this.
    TARGET: Il n'y avait pas un mot à répondre.
 PREDICTED: Il n ' y avait pas de répondre .
--------------------------------------------------------------------------------
    SOURCE: The curtain went up. I have often seen Marguerite at the theatre.
    TARGET: Elle crut s'être trompée et détourna la tête. On leva le rideau.
 PREDICTED: Le rideau se mit à voir . Je n ' ai vu le théâtre à Marguerite .
--------------------------------------------------------------------------------
Epoch 05: 100%|██████████| 966/966 [03:04<00:00,  5.24it/s, loss=3.314]
Average loss of epoch 5 is 3.2554558915381104
--------------------------------------------------------------------------------
    SOURCE: 'I am certain of it,' Julien at once rejoined.
    TARGET: – J’en suis sûr, répliqua vivement Julien.
 PREDICTED: – Je suis certain , dit Julien en s ’ éloigna .
--------------------------------------------------------------------------------
    SOURCE: The second distance between the stick and the bottom of the cliff was five hundred feet.
    TARGET: La deuxième distance, entre le piquet et la base de la muraille, était de cinq cents pieds.
 PREDICTED: La seconde partie entre le fond de la muraille et la muraille de cinq cents pieds .
--------------------------------------------------------------------------------
Epoch 06: 100%|██████████| 966/966 [03:03<00:00,  5.26it/s, loss=3.016]
Average loss of epoch 6 is 3.168518413174473
--------------------------------------------------------------------------------
    SOURCE: Do you still miss your Cubans, sir?"
    TARGET: Regrettez-vous les londrès, monsieur ?
 PREDICTED: Vous avez toujours votre , monsieur ?
--------------------------------------------------------------------------------
    SOURCE: Ah! my Lord God!
    TARGET: Ah ! mon Dieu Seigneur !
 PREDICTED: Ah ! mon Dieu !
--------------------------------------------------------------------------------
Epoch 07: 100%|██████████| 966/966 [03:02<00:00,  5.28it/s, loss=3.126]
Average loss of epoch 7 is 3.0993083898078333
--------------------------------------------------------------------------------
    SOURCE: I had finished: Miss Temple regarded me a few minutes in silence; she then said--
    TARGET: J'avais achevé; Mlle Temple me regarda en silence pendant quelques minutes; puis elle me dit:
 PREDICTED: Je fus terminé : Mlle Temple me parut quelques minutes ; puis elle dit :
--------------------------------------------------------------------------------
    SOURCE: One is expected to be a complete nonentity, and at the same time give no one any grounds for complaint.
    TARGET: Il faudrait y être d’une nullité parfaite, et cependant ne donner à personne le droit de se plaindre.
 PREDICTED: On ne s ' attendait pas d ' un , et , à la même fois , on ne fait aucune trace .
--------------------------------------------------------------------------------
Epoch 08: 100%|██████████| 966/966 [03:05<00:00,  5.22it/s, loss=3.038]
Average loss of epoch 8 is 3.045017740000849
--------------------------------------------------------------------------------
    SOURCE: This matter of fuelling steamers is a serious one at such distances from the coal-mines; it costs the Peninsular Company some eight hundred thousand pounds a year.
    TARGET: Grave et importante affaire que cette alimentation du foyer des paquebots à de telles distances des centres de production. Rien que pour la Compagnie péninsulaire, c'est une dépense annuelle qui se chiffre par huit cent mille livres (20 millions de francs).
 PREDICTED: Cette affaire est de plus de , en un tel sont les mines de charbon ; il est le train de huit cents livres .
--------------------------------------------------------------------------------
    SOURCE: Après avoir salué respectueusement la marquise, Gonzo ne s’éloigna point comme de coutume pour aller prendre place sur le fauteuil qu’on venait de lui avancer.
    TARGET: After respectfully greeting the Marchesa, Gonzo did not withdraw as usual to take his seat on the chair which had just been pushed forward for him.
 PREDICTED: After my first , the Marchesa must be able to be able to be , as to be , where a face of him , in his .
--------------------------------------------------------------------------------
Epoch 09: 100%|██████████| 966/966 [03:04<00:00,  5.25it/s, loss=3.133]
Average loss of epoch 9 is 3.0063985633060306
--------------------------------------------------------------------------------
    SOURCE: He found also, on touching them that these guns were breech-loaders.
    TARGET: Il vérifia même, en les touchant, que ces canons se chargeaient par la culasse.
 PREDICTED: Il y avait aussi , sur les gros canons qui le de .
--------------------------------------------------------------------------------
    SOURCE: The happiness anticipated by Catherine and Lydia depended less on any single event, or any particular person, for though they each, like Elizabeth, meant to dance half the evening with Mr. Wickham, he was by no means the only partner who could satisfy them, and a ball was, at any rate, a ball.
    TARGET: La joie que se promettaient Catherine et Lydia dépendait moins de telle personne ou de telle circonstance en particulier ; bien que, comme Elizabeth, chacune d’elles fut décidée a danser la moitié de la soirée avec Mr. Wickham, il n’était pas l’unique danseur qui put les satisfaire, et un bal, apres tout, est toujours un bal.
 PREDICTED: Le bonheur de la bonheur , Lydia et Lydia n ’ en était pas moins d ’ un autre chose , mais Elizabeth , comme pour Elizabeth , comme pour Elizabeth , pour avoir la veille .
--------------------------------------------------------------------------------
```

## Results
Achieved loss = 3.1 in 10 epochs.

## Reference:
1. Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates
2. Lessons on Parameter Sharing across Layers in Transformers
