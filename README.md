firstly train the autoencoder \
  python path/to/reconstruction_project/autoencoder/net1/autoencoder.py path/to/database \
  python path/to/reconstruction_project/autoencoder/net2/autoencoder.py path/to/database \
 
the model in net2 is what we need.\

use the program for model reconstruction.\
  sastbx.python path/to/reconstruction_project/reconstrcution/main.py \
  --iq_path path/to/iq_file --rmax radius/of/model --output_folder path/to/place/result --target_pdb path/to/target/model \
  
  the target_pdb is not necessary. If you provide, the program will make a comparison between the reconstructed model and the target model. \

after running the program, you will get the result in the output_folder in give. \

there are several import files：\
    out.pdb, score_mat.txt, cc_mat.txt \
    out.pdb is the final reconstructed model. \
    score_mat.txt records the score of top20 objects in every generation in our reconstruction method. the score is the distance between the iq curve and the target iq file.
    cc_mat.txt will be generated when you provide the target_pdb. it records the correlation between the top20 object of every generation and the target object.
    the column for the score_mat.txt and the cc_mat.txt is both 20. the rows is the number of total iterations.
