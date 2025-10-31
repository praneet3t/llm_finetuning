from unsloth import FastLanguageModel
from transformers import GenerationConfig
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import json
import torch

# 1. Load model with Unsloth
base_model = "unsloth/Llama-3.2-3B-Instruct"  # or phi-2 etc.
adapter_path = "/home/incois/tvsubhaskar/llm_project/LLaMA-3.2-3B-Instruct/finetuned_model"

max_seq_length = 2048
dtype = torch.float16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = True,
)

model = FastLanguageModel.load_adapter(model, adapter_path)

# 2. Evaluation Questions and Ground Truths
questions_and_answers = [
    {"question": "Evaluate the long-term implications of rising sea surface temperature on global fishery migration patterns.", "answer": "Rising sea surface temperatures are causing shifts in the distribution of fish stocks, often pushing them toward cooler, deeper, or more polar waters. This disrupts traditional fishing zones, affects international fishing agreements, and can cause economic imbalances in developing regions dependent on fishery exports."},
    
    {"question": "Critically assess how underwater volcanic activity can influence large-scale ocean circulation patterns.", "answer": "Underwater volcanic activity introduces heat and mineral-rich plumes that can alter local density gradients, which in turn impact thermohaline circulation over time. Persistent eruptions could even slightly shift deep water formation zones, subtly influencing long-term oceanic conveyor belts."},
    
    {"question": "Predict the consequences of declining Argo float coverage in the Indian Ocean on marine weather forecasting accuracy.", "answer": "A reduction in Argo float coverage would result in decreased real-time profiling of ocean temperature and salinity, thereby degrading model inputs for coupled atmosphere-ocean systems. This can lead to inaccurate forecasts of monsoons, cyclones, and upwelling phenomena critical for coastal planning."},
    
    {"question": "Compare the impact of wind-driven currents versus thermohaline circulation on nutrient distribution in the upper ocean layers.", "answer": "Wind-driven currents dominate short-term nutrient redistribution in surface layers, especially in coastal upwelling zones, whereas thermohaline circulation influences deep nutrient replenishment on longer temporal scales, sustaining global biological productivity."},
    
    {"question": "Discuss the effectiveness of satellite altimetry in monitoring mesoscale eddies compared to in-situ buoy arrays.", "answer": "Satellite altimetry offers broad spatial and temporal coverage ideal for tracking mesoscale eddies across ocean basins, whereas buoy arrays provide high-resolution local data on vertical structure. A combination yields optimal eddy characterization."},
    
    {"question": "Evaluate the role of ocean salinity anomalies in modulating the intensity of El Niño events.", "answer": "Salinity anomalies influence stratification and upper ocean heat content. Fresher surface waters suppress vertical mixing, trapping heat, and potentially amplifying El Niño intensity when coupled with favorable atmospheric conditions."},
    
    {"question": "Examine how swell propagation from distant storms influences port operations and coastal erosion in semi-enclosed seas.", "answer": "Swell energy can travel thousands of kilometers, impacting semi-enclosed seas by increasing wave setup near shores, disrupting docking schedules, and accelerating beach erosion despite local calm weather conditions."},
    
    {"question": "Infer the potential risks posed by illegal trawling on submarine cable infrastructure in high-traffic fishing zones.", "answer": "Illegal trawling activities often ignore regulated cable protection zones, increasing the likelihood of accidental cable dragging or severing. Such damage can lead to costly communication outages and delay emergency marine operations."},
    
    {"question": "Justify the inclusion of biogeochemical sensors on modern Argo floats for climate change studies.", "answer": "Biogeochemical sensors allow Argo floats to capture oxygen, pH, nitrate, and chlorophyll data, essential for understanding ocean carbon cycles, acidification trends, and primary productivity shifts under climate forcing scenarios."},
    
    {"question": "Discuss the feedback loop between Arctic sea ice melt and changes in the Atlantic Meridional Overturning Circulation (AMOC).", "answer": "Arctic ice melt dilutes North Atlantic surface salinity, reducing density and slowing down the sinking branch of AMOC. A weaker AMOC further warms the Arctic, exacerbating ice loss and reinforcing the feedback loop."},
    
    {"question": "Assess the vulnerability of coral reef systems to internal wave dynamics in narrow continental shelves.", "answer": "Internal waves can bring cooler, nutrient-rich waters to coral ecosystems, buffering thermal stress. However, in narrow shelves, intensified wave action can cause abrupt temperature drops and nutrient surges, stressing coral physiology and increasing bleaching risk."},
    
    {"question": "Evaluate the role of coastal Kelvin waves in early warning systems for tsunamis and storm surges.", "answer": "Kelvin waves, constrained by coastlines, propagate rapidly and can serve as precursors to incoming disturbances. Monitoring their anomalies enhances prediction models for storm surges and near-shore tsunami amplification."},
    
    {"question": "Critically assess the reliability of sea surface height anomalies in forecasting equatorial Kelvin wave propagation.", "answer": "Sea surface height anomalies derived from satellite altimetry are a robust proxy for tracking Kelvin waves due to their associated thermocline displacement. However, accuracy degrades in regions with sparse calibration points or strong cloud interference."},
    
    {"question": "Infer how mesoscale eddies influence the vertical flux of microplastics in oligotrophic gyres.", "answer": "Mesoscale eddies enhance vertical mixing and can trap or redistribute microplastics both horizontally and vertically. In oligotrophic regions, this mixing disrupts stratification, allowing microplastics to penetrate deeper into the water column and affect pelagic organisms."},
    
    {"question": "Predict the operational challenges posed by sudden changes in mixed layer depth for autonomous underwater vehicles (AUVs).", "answer": "Fluctuating mixed layer depths can alter acoustic properties, current velocities, and thermal gradients, impacting AUV buoyancy control, navigation accuracy, and energy consumption during missions."},
    
    {"question": "Compare the utility of scatterometer wind data with buoy-based measurements in open ocean forecasting.", "answer": "Scatterometer data provides global wind vector fields useful for synoptic-scale modeling, whereas buoys offer localized, high-frequency data critical for calibration and real-time event validation. Both are complementary."},
    
    {"question": "Examine the potential feedbacks between ocean deoxygenation and greenhouse gas emissions from marine ecosystems.", "answer": "Ocean deoxygenation alters microbial community structures, increasing anaerobic pathways that emit N₂O and CH₄. These potent greenhouse gases further warm the planet, intensifying deoxygenation through stratification."},
    
    {"question": "Discuss the implications of swell refraction around submerged atolls for navigation safety.", "answer": "Swell refraction causes wave energy to bend around submerged features, focusing on certain zones unpredictably. This can create localized high-energy areas that threaten small vessels and complicate navigation charts."},
    
    {"question": "Evaluate the effect of internal tides on vertical nutrient transport in the Bay of Bengal.", "answer": "Internal tides break near continental slopes, driving vertical nutrient flux into the euphotic zone. This enhances regional productivity despite overall weak surface turbulence due to strong stratification."},
    
    {"question": "Critically evaluate how AIS data anomalies might reveal illegal transshipments in high seas.", "answer": "Anomalies such as sudden AIS blackouts, loitering patterns, or synchronized paths between vessels in remote areas can indicate unauthorized transshipments, although false positives must be filtered through trajectory and environmental context."},
    
    {"question": "Compare how wind-wave misalignment affects wave energy dissipation in coastal versus open-ocean regions.", "answer": "In coastal zones, wind-wave misalignment increases bottom friction and topographic interactions, rapidly dissipating energy. In the open ocean, dispersion occurs more gradually, affecting swell coherence over long distances."},
    
    {"question": "Examine the correlation between MJO (Madden-Julian Oscillation) phases and cyclone genesis in the Indian Ocean.", "answer": "Active MJO phases with enhanced convection over the Indian Ocean increase atmospheric instability and vorticity, creating favorable conditions for tropical cyclone formation, especially in pre- and post-monsoon periods."},
    
    {"question": "Assess how wave-current interactions distort sea surface roughness seen in SAR satellite images.", "answer": "Wave-current interactions can enhance or suppress surface roughness, altering backscatter signatures in SAR data. This complicates interpretation unless current vectors are accounted for in preprocessing algorithms."},
    
    {"question": "Discuss how stochastic wave models can improve ensemble forecasting of rogue waves.", "answer": "Stochastic models account for probabilistic interference of multiple wave trains, enabling ensemble forecasts to estimate rogue wave likelihood more realistically than deterministic approaches."},
    
    {"question": "Justify the relevance of coupling sediment transport models with hydrodynamic simulators in estuarine systems.", "answer": "Coupling allows better prediction of morphological changes in response to tides, storms, and anthropogenic modifications. It supports flood risk assessments and navigational safety planning in estuaries."},
    
    {"question": "Infer the impacts of glacial meltwater input on fjord stratification and mixing processes.", "answer": "Glacial melt introduces low-salinity surface water, strengthening stratification and reducing vertical mixing. This alters nutrient dynamics, affects primary productivity, and can trap heat, impacting fjord ecology."},
    
    {"question": "Critically assess the use of wavelet transforms in detecting non-stationary oscillations in sea surface height data.", "answer": "Wavelet transforms excel at isolating transient, localized frequency signals in sea surface height data, useful for identifying Kelvin waves or tsunami precursors. However, selection of wavelet basis and scale resolution affects accuracy."},
    
    {"question": "Predict how future Arctic shipping routes will be affected by changing sea-ice extent variability.", "answer": "Decreasing sea-ice extent opens new seasonal routes but increases unpredictability due to higher interannual variability and ice drift hazards. Dynamic routing systems with real-time satellite input will be essential for safety."},
    
    {"question": "Evaluate the ecological consequences of artificial upwelling systems deployed in nutrient-poor gyres.", "answer": "Artificial upwelling introduces nutrients into oligotrophic regions, stimulating primary production. While potentially aiding carbon sequestration, it may disrupt native ecosystems, encourage harmful algal blooms, or alter food web dynamics."},
    
    {"question": "Compare the performance of machine learning-based wave forecasting models with traditional spectral methods.", "answer": "Machine learning models, when trained on large datasets, can outperform spectral methods in short-term, nonlinear wave predictions, but may lack physical interpretability and generalization to unseen environmental regimes."}
]

# 3. Metric Functions
def compute_metrics(prediction, reference):
    smoothie = SmoothingFunction().method4

    bleu = sentence_bleu(
        [reference.split()],
        prediction.split(),
        smoothing_function=smoothie
    )

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, prediction)
    rouge_l = rouge_scores['rougeL'].fmeasure

    # F1 Score
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(ref_tokens) if ref_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    # Exact Match
    em = int(prediction.strip().lower() == reference.strip().lower())

    return {
        "BLEU": round(bleu, 4),
        "ROUGE-L": round(rouge_l, 4),
        "F1": round(f1, 4),
        "Exact Match": em
    }

# 4. Run Evaluation
model.eval()
results = []
for i, qa in enumerate(questions_and_answers):
    prompt = f"### Question:\n{qa['question']}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_answer = decoded.split("### Answer:")[-1].strip()

    metrics = compute_metrics(generated_answer, qa['answer'])
    results.append({
        "Q#": i+1,
        "Question": qa['question'],
        "Reference": qa['answer'],
        "Prediction": generated_answer,
        **metrics
    })

# 5. Save Results
with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Print summary
for res in results:
    print(f"Q{res['Q#']}: BLEU={res['BLEU']}, ROUGE-L={res['ROUGE-L']}, F1={res['F1']}, EM={res['Exact Match']}")
