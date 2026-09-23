# Scope and Marketing Notes

## Where we're starting

We're not building "an English app." We're building a coach you can actually call. Someone opens the app, picks a real IELTS Speaking topic, and has a short spoken conversation with an AI coach, close to what the real IELTS Speaking test feels like. When the call ends, they get an estimated band score broken down across fluency, vocabulary, grammar and pronunciation, plus two or three concrete fixes they can act on immediately.

That's version one. No writing or listening modules, no open ended free chat. The call itself follows the real IELTS Speaking format: one fixed topic or cue card, a short guided back and forth, then it ends and the score comes back. To keep this buildable for two people, we're not building the voice pipeline from scratch. Tools like Vapi, Retell or ElevenLabs' conversational API already handle the speech to text, the model's turn, and the voice response, so the engineering work is mostly wiring that up and building the scoring layer on top of it.

## Where we expand once this works

Once IELTS Speaking has real users and the scores hold up against actual IELTS results, the natural next steps are:

**PTE Academic speaking.** Same core engine, different question format. Big audience, especially for Australia and UK visa routes.

**TOEFL Speaking.** Popular for US university admissions, slightly smaller community online than IELTS but still meaningful.

**Duolingo English Test (DET).** Growing fast, cheaper and quicker than the traditional exams, and a lot of universities now accept it. Less competition here than in the IELTS tool space.

We hold off on all three until the first product actually earns trust. A bad score on someone's first try kills word of mouth before it starts.

## Subreddits to use for marketing

**r/IELTS.** This is the main one. Around 100,000 members and very active, with "Speaking" as one of the most discussed topics. People post things like "why did I get 5.5 for Speaking" constantly. It's also heavily moderated by actual certified IELTS teachers, so a link drop post will get pulled fast. This one needs genuine participation first, promotion later.

**r/ieltsspeaking.** Smaller, but exactly on topic. Worth checking their rules before posting since it's a niche community that notices when someone shows up only to sell something.

**r/IELTS_Guide.** Around 13,000 members. This one is restricted, meaning only approved users can start posts, but others can comment. Good place to build a reputation in the comments even if you can't post directly at first.

**r/ieltswriting.** About 7,000 members and technically about writing, but a lot of crossover with speaking questions, and self promotion posts do appear here and survive, which tells us the mods are more relaxed about it than the main sub.

**r/TOEFL** and **r/ToeflAdvice.** TOEFL's main sub is small, a couple thousand members, but ToeflAdvice has grown to around 30,000 and is adding members quickly. Good for when we expand into TOEFL.

**r/ToeflSpeaking.** Small and quiet right now, but exactly the right audience when we get there.

**r/learnEnglishOnline** and **r/LearningEnglish.** Around 50,000 and 30,000 members. Broader than exam prep specifically, but full of people trying to improve spoken English generally, and LearningEnglish in particular is growing very fast.

**r/languagelearning.** Huge, over 3 million members, but general purpose. Not exam focused, so posts here need to be framed around fluency and confidence rather than band scores. Good for reach once we have something people naturally want to share.

**PTE communities on Reddit** exist but are smaller and messier than the IELTS ones, with a fair amount of spam and off topic posts. Worth watching once we build the PTE version, not worth heavy investment right now.

Outside Reddit, there's also a large IELTS Prep Discord server, around 36,000 members, built specifically around people pairing up to practice speaking with each other. That's arguably an even better fit than Reddit itself since the entire server exists for this one purpose.

## How we actually show up

Every one of these communities has seen people try to drop a link and disappear, and they've gotten good at ignoring or removing that. The approach that works is answering real "rate my speaking" and "am I band 7" posts with genuine, specific feedback for a couple of weeks before we ever mention the tool. By the time we do post something, we're a known, helpful person in the community, not a stranger selling software.