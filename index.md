# Thermal Spread Functions (TSF): Physics-guided Material Classification

<div class="btn-group btn-group-lg">
  <a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Dashpute_Thermal_Spread_Functions_TSF_Physics-Guided_Material_Classification_CVPR_2023_paper.pdf" class="btn btn-primary" role="button" target="_blank">PDF</a>
  <a href="https://github.com/aniketdashpute/TSF" class="btn btn-primary" role="button" target="_blank">Code</a>
  <a href="https://openaccess.thecvf.com/content/CVPR2023/supplemental/Dashpute_Thermal_Spread_Functions_CVPR_2023_supplemental.pdf" class="btn btn-primary" role="button" target="_blank" >Supplementary Material</a>
</div>

------------------------------

### Thermal Spread Function (TSF)

- Heating-Cooling creates unique spatio-temporal curves that are unique to each material

<p float="middle">
  <img src="https://github.com/aniketdashpute/TSF/assets/52461513/041d3de1-faf4-4059-8aca-8082ad814138" width = 500/>
</p>

### Finite Differences Approach for the Inverse Problem

- We extract the diffusivity and absorption parameters from the TSFs. These form the basis for the classification

<p float="middle">
  <img src="https://github.com/aniketdashpute/TSF/assets/52461513/5b42c069-efa1-4bbd-a591-afbd5905c609" width = 600/>
</p>

### Classification based on recovered thermal properties

- We use the extracted thermal properties to train an MLP-based classifier for identifying a new material

<p float="middle">
  <img src="https://github.com/aniketdashpute/TSF/assets/52461513/134a2019-9d4c-46a9-b766-5324d1d86f20" width = 600/>
  <img src="https://github.com/aniketdashpute/TSF/assets/52461513/3933f506-34b9-4393-af4d-357d4ddf79c7" width = 600/>
</p>


------------------------------

Aniket Dashpute<sup>1,3</sup>, Vishwanath Saragadam<sup>3</sup>, Emma Alexander<sup>2</sup>, Florian Willomitzer<sup>4</sup>, Aggelos Katsaggelos<sup>1</sup>, Ashok Veeraraghavan<sup>3</sup>, Oliver Cossairt<sup>1,2</sup>

<sup>1</sup>Electrical and Computer Engineering, <sup>2</sup>Computer Science, Northwestern University,

<sup>3</sup>Electrical and Computer Engineering, Rice University,

<sup>4</sup>Wyant College of Optical Sciences, University of Arizona

------------------------------

<div class="links">
      <a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Dashpute_Thermal_Spread_Functions_TSF_Physics-Guided_Material_Classification_CVPR_2023_paper.pdf" class="btn btn-lg z-depth-0" role="button" target="_blank" style="font-size:12px;">PDF</a>
      <a href="https://github.com/aniketdashpute/TSF" class="btn btn-lg" role="button" target="_blank" style="font-size:12px;">Code</a>
      <a href="https://aniketdashpute.github.io/TSF/" class="btn btn-lg" role="button" target="_blank" style="font-size:12px;">Project Page</a>
</div>

