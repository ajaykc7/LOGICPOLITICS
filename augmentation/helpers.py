# APPEALING TO EMOTION
emotion = '''
Appealing to emotion, a fallacy category, is described as "manipulation of the recipient’s emotions in order to win an argument". Its loose logical form is: "[CLAIM] is made without evidence. In place of evidence, emotion is used to convince the interlocutor that [CLAIM] is true". Your task is to make a political-related sentence to become fallacious, using the provided description and the loose logical form. Based on examples #1 to #7 below, please complete a fallacy for sentence #8:


Source \#1: Illegal immigration hurts the economy

Fallacy \#1: "There's a direct connection between illegal immigration and the downfall of the economy. Just look at how it lead to increase in crime, the world has never been this dangerous before!"

Source \#2: Global integration exposes deep fault lines in the existing international order.

Fallacy \#2: "Let's not forget the history of global integration. It's always been about exploitation, and the exploitation of our resources will always come at a cost."

Source \#3: We expect the United Nations to play an effective and vital role in resolving conflicts.

Fallacy \#3: "The United Nations is the only hope for peace in the world. We can't afford to let it fail us. Don't you care about the safety and security of your children and grandchildren?"

Source \#4: Citizens are demanding respect for the dignity of all people.

Fallacy \#4: "Respect for the dignity of all people is a threat to the safety and security of our nation. We can't afford to entertain such demands, especially when they come from those who don't share our values."

Source \#5: Global warming is not real

Fallacy \#5: "Nature has its own way of balancing things out, and global warming is nothing more than just part of a natural cycle."

Source \#6: Globalization, while raising global living standards, has weakened workers’ positions and their ability to earn a decent wage.

Fallacy \#6: "Globalization may have brought some benefits, but it has also led to the exploitation of workers and their inability to earn a decent wage. Don't you care about the future of our economy and the well-being of our people?"

Source \#7: Advancements in science and technology, along with political changes, have significantly improved the quality of life for people globally and increased access to knowledge and freedom.

Fallacy \#7: "But at what cost? Every new breakthrough in technology brings us closer to a world where our privacy is constantly violated."

Source \#8:'''

# FALSE DILEMMA
dilemma = '''
False dilemma, a fallacy category, is described as "A claim presenting only two options or sides when there are many options or sides". Your task is to make a political-related sentence to become fallacious, using the provided description and the loose logical form. Based on examples #1 to #4 below, please complete a fallacy for sentence #5:

Source #1: Global warming is not real.
Fallacy #1: "Global warming can't be a real thing. Just look at how nature has its own way of balancing things out throughout the history of the universe."
Source #2: America's potential as a net energy exporter in the near future should not be jeopardized.
Fallacy #2: "America is poised to become a net energy exporter over the next decade. We should not abandon that progress at the cost of weakening our energy renaissance and crippling economic growth."
Source #3: Taking action to address climate change is important.
Fallacy #3: "I don’t want to give up my car, so I don’t think I can support fighting climate change."
Source #4: Illegal immigration hurts the economy.
Fallacy #4: "There are only two options when it comes to illegal immigration - either we allow it and harm the economy or we deport everyone and disrupt families."
Source #5:'''

# FALLACY OF RELEVANCE
relevancy = '''
Fallacy of Relevance, also known as red herring, occurs when the speaker attempts to divert attention from the primary argument by offering a point that does not suffice as counterpoint/supporting evidence (even if it is true). Its loose logical form is: "It is claimed that [MSK1] implies [MSK2], whereas [MSK1] is unrelated to [MSK2]". Your task is to make a source sentence to become fallacious, using the provided description and the loose logical form. Based on examples #1 to #7, please complete a fallacy for sentence #8:

Source #1: Citizens are demanding respect for the dignity of all people.
Fallacy #1: "We need to focus more on national security and economic growth. The respect for the dignity of all people is a luxury that is only affordable after a nation is stable and prosperous."
Source #2: We cannot trust the government. They have been making so many mistakes that have jeopardized our rights and freedoms.
Fallacy #2: "There's no reason to doubt the government’s actions. People are not capable of defending themselves either, they have no right to criticize."
Source #3: Globalization has weakened workers’ positions and their ability to earn a decent wage.
Fallacy #3: "Globalization has increased access to a wider range of products and services, which is beneficial for consumers. So it should not be a problem even though it negatively affects people's income."
Source #4: Illegal immigration hurts the economy.
Fallacy #4: "People that illegally migrate to our country use more public services than they pay in taxes, thus they have bad impact on the economy."
Source #5: There's a big concern on the financial burden from relocating the office to California.
Fallacy #5: "The weather in California is so much warmer, we must move the office there."
Source #6: Several employees were dissapointed because they weren't promoted as promised by the compaany.
Fallacy #6: "It's okay for the company to not raise salaries since they still provide great benefits for the employees."
Source #7: During the politician's tenure, the party was heavily corrupted.
Fallacy #7: "There was a little issue of corruption last year, but look at how much more corrupt they are in the other party!""
Source #8:'''

# INTENTIONAL FALLACY
intention ='''
Intentional fallcy is a fallacy category for when an argument has some element that shows intent of a speaker to win an argument without actual supporting evidence. Its loose logical form is: "[MSK1] knows [MSK2] is incorrect. [MSK1] still claim that [MSK2] is correct using an incorrect argument." Your task is to make a source sentence to become fallacious, using the provided description and the loose logical form. Based on examples #1 to #2, please complete a fallacy for sentence #3:

Source \#1: The government has been withholding the information about extraterrestrials existence from us for its own interests.

Fallacy \#1: “No one has ever been able to prove that extraterrestrials exist, so they must not be real."

Source \#2: Using positive discipline builds better connections with kids, questioning the idea that hitting them is both effective and ethical.

Fallacy \#2: "It’s common sense that if you smack your children, they will stop the bad behavior. So don’t tell me not to hit my kids."

Source \#3:'''


def make_prompts(sentence_list, fallacy):
    """
    Parameters
    ----------
    sentences : str
        List of compact sentences to build prompt for
    fallacy:
        Fallacy-specific prompt, with form:
        
        <fallacy description>, <loose logical form>, <task description>
        Source #1: <sample source 1>
        Fallacy #1: <sample corresponding fallacy 1>
        Source #2: <sample source 2>
        Fallacy #2: <sample corresponding fallacy 2>
        ...
        Source #k:
    
    Return
    ------
    prompts:
        Prompt dataset, each entry sentence[i] has the form:
        
        <fallacy description>, <loose logical form>, <task description>
        Source #1: <sample source 1>
        Fallacy #1: <sample corresponding fallacy 1>
        Source #2: <sample source 2>
        Fallacy #2: <sample corresponding fallacy 2>
        ...
        Source #k: <sentence[i]>
        Fallacy #k:
    """

    prompts = []
    for i in sentence_list:
        i = i.replace("\n", " ")
        prompt = "[INST] " + fallacy + " " + i + f"\nFallacy #{int(fallacy[-2])}: " + "\n[/INST]\n"
        prompts.append({"source": i, "prompt": prompt})
    
    return prompts