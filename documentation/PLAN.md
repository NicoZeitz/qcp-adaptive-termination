I need a bigger architecture rework to support the nestest features and so on. here is what should be changed

1. Remove langchang prompt management
* Use chevron (mustache python library) instead
* no loading prompts from langchain
* load prompts from pompt folder instead
* change structure in prompt folder to have no folder, instead longer filenames like discovery-coordinator-relevance-instructions.md
   * query coordinator -> coordinator, sql data agent -> data-agent-sql
   * stages: discovery, planning, execution
   * 0-system -> instructions
   * 1-user -> keep like it is
* still keep langfuse for exporter

2. remove text data agent as it is not needed anymore

3. add a new method next to run_agent, which is called run_llm which is just a normal llm call. this should just call run agent, but validate a few things before (no tools, set turns to 1)

4. to code restructuring at this stage if necessary

5. Take a look at the documentation folder implement the changes listed there. All of them. Mainly orient yourself at the svg files. also restructure the code if necessary. Also ask me questions in case of any questions or uncertainties. Each LLM/Agent Call should have it's own distinct prompt. In case of the coordinator schema alignmnet task it may need two difference ones based on if it is the first iteration or a later one.
Also decide how to deal with errors at each stage. Useless possibilities like evaluator wanting to query more agents but all agents are already queries should not be possible.
Also restrict the json schema of a plan (like it is similarilty done right know) to only allow agents in multi agent query step and not the coordinator
Modify all the prompts correspondenly as well (if needed)
Generally keep the architecture open to further modifications (for example that the result list for many agents can later also propagate a uncertainty estimate for the value column)

6. Change the code a bit more so that
* the completness evaluator still decides if more agents should be queries or a result list is returned.
* in case a result list is returned, but more agents are still available go into the query loop again. Record the interactions in each loop and at the end return the result list. We want to be able to plot a query more agents vs more accurate result plotting possible

Pause after each step and ask me first if the result is ok, so i can take a look at it