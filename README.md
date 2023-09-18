# Argumentation Models

The models included in this repository are implementations of the argumentation frameworks I developed as part of the dissertation I wrote on creating a 'System To Support A Group Of Humans In Making Joint Decisions With Arguments' whilst studying for my Artifical Intelligence MSc at King's College London.  This dissertation is available within the repository.

Each file contains detailed instructions on how to create and evaluate the different argumentation frameworks.  In each instance, users will need to define the abstract argumentation framework by specifying the arguments, the attacks and the supports.  Depending on the framework being used, the user will then need to input the relevant subjective input for each of the arguments.  These are as follows:

i) Preference-Based Bipolar Argumentation Framework ('Pref-BAF'): a preference ordering between all the arguments for each participant

ii) Value-Based Bipolar Argumentation Framework ('Val-BAF'): the corresponding social vaulue for each argument, and a preference ordering between these values for
each partipant

iii) Voting-Based Bipolar Framework ('Vote-BAF'): a weighted vote (eg from 1-6) for each argument for each participant

Once the framework and subjective input has been entered, then running the code will evaluate the framework and return the corresponding argumentation graph.  

Further details on how to implement each model are contained in the files.
