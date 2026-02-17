# Done

python aya_dataset.py --push_only
python common_corpus.py --push_only
python common_pile.py --all --push_only
python eurovoc.py --push_only

python gallica.py --name monographies --push_only
python gallica.py --name press --push_only
python kurakurai_scholar.py --push_only
python hal_cea.py --push_only

# stem, chat, code, math => see what have been done.
python nemotron_posttraining.py --subset stem --push_only
python nemotron_posttraining.py --subset chat --push_only
python nemotron_posttraining.py --subset code --push_only
python nemotron_posttraining.py --subset math --push_only
python nemotron_posttraining.py --subset multilingual_fr --push_only
python nemotron_posttraining.py --subset multilingual_es --push_only
python nemotron_posttraining.py --subset multilingual_it --push_only
python nemotron_posttraining.py --subset multilingual_de --push_only

python nemotron_posttraining.py --keep_thinking --subset stem --push_only
python nemotron_posttraining.py --keep_thinking --subset chat --push_only
python nemotron_posttraining.py --keep_thinking --subset code --push_only
python nemotron_posttraining.py --keep_thinking --subset math --push_only
python nemotron_posttraining.py --keep_thinking --subset multilingual_fr --push_only
python nemotron_posttraining.py --keep_thinking --subset multilingual_es --push_only
python nemotron_posttraining.py --keep_thinking --subset multilingual_it --push_only
python nemotron_posttraining.py --keep_thinking --subset multilingual_de --push_only

python nemotron_posttraining_translation.py --subset multilingual_fr --push_only

python paradocs.py --languages en-fr --revert 0.5 --push_only
python paradocs.py --languages en-es --revert 0.5 --push_only
python paradocs.py --languages en-nl --revert 0.5 --push_only
python paradocs.py --languages en-de --revert 0.5 --push_only 
python paradocs.py --languages en-pt --revert 0.5 --push_only
python paradocs.py --languages en-it --revert 0.5 --push_only

python gutenberg.py --push_only

python lucie_dataset.py --push_only --name CroissantAligned
python lucie_dataset.py --push_only --name MathPile
python lucie_dataset.py --push_only --name Youtube
python lucie_dataset.py --push_only --name Theses
python lucie_dataset.py --push_only --name InterventionsParlement
python lucie_dataset.py --push_only --name QuestionsEcritesParlement
python lucie_dataset.py --push_only --name DiscoursPublics
python lucie_dataset.py --push_only --name AmendementsParlement

python finemath.py --name finemath-3plus --push_only
python finemath.py --name infiwebmath-3plus --push_only

python wikimedia.py --push_only
python vikidia.py --push_only

python culturax.py --language fr --push_only
python hplt2.py --language fra_Latn --push_only
python fineweb2.py --language fra_Latn --push_only

python fineweb2.py --language acf_Latn --push_only
python fineweb2.py --language arb_Arab --push_only
python fineweb2.py --language bre_Latn --push_only
python fineweb2.py --language cat_Latn --push_only
python fineweb2.py --language cos_Latn --push_only
python fineweb2.py --language crs_Latn --push_only
python fineweb2.py --language deu_Latn --push_only
python fineweb2.py --language eus_Latn --push_only
python fineweb2.py --language frp_Latn --push_only
python fineweb2.py --language gcf_Latn --push_only
python fineweb2.py --language gcr_Latn --push_only
python fineweb2.py --language ita_Latn --push_only
python fineweb2.py --language nld_Latn --push_only
python fineweb2.py --language oci_Latn --push_only
python fineweb2.py --language pcd_Latn --push_only
python fineweb2.py --language por_Latn --push_only
python fineweb2.py --language rcf_Latn --push_only
python fineweb2.py --language spa_Latn --push_only
python fineweb2.py --language tah_Latn --push_only
python fineweb2.py --language wln_Latn --push_only

python fineweb2_hq.py --language arb_Arab --push_only --force 
python fineweb2_hq.py --language deu_Latn --push_only --force 
python fineweb2_hq.py --language fra_Latn --push_only --force 
python fineweb2_hq.py --language ita_Latn --push_only --force 
python fineweb2_hq.py --language nld_Latn --push_only --force 
python fineweb2_hq.py --language por_Latn --push_only --force 
python fineweb2_hq.py --language spa_Latn --push_only --force 

python insee.py 
python opendata.py --push_only
python europarl.py
python fineweb_edu.py --push_only
python dclm.py --push_only
python starcoder.py --push_only
python olmo_mix.py --name starcoder --push_only

python claire.py --language fr --push_only
python claire.py --language en --push_only

# Phase 1/2
megamath

## robots.txt
python get_robots_txt.py --push_only
