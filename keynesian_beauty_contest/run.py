import logging
import asyncio
import math
import time
import os
from naptha_sdk.user import sign_consumer_id, get_private_key_from_pem
from naptha_sdk.modules.agent import Agent
from naptha_sdk.schemas import OrchestratorRunInput, OrchestratorDeployment, AgentRunInput
from keynesian_beauty_contest.schemas import InputSchema
from typing import Dict, List

logger = logging.getLogger(__name__)

class KeynesianBeautyContest:
    """Keynesian beauty contest orchestrator implementation with batched execution"""
    def __init__(self, orchestrator_deployment: OrchestratorDeployment, *args, **kwargs):
        self.orchestrator_deployment = orchestrator_deployment
        logger.info(f"Orchestrator deployment initialized")
        self.agent_deployments = self.orchestrator_deployment.agent_deployments
        logger.info(f"Found {len(self.agent_deployments)} agent deployments")
        
    async def run_single_agent(self, agent_index: int, node_index: int, consumer_id: str):
        """Run a single agent and handle errors"""
        name = f"Agent_{agent_index}"
        logger.info(f"Preparing {name} on node {node_index}")
        
        try:
            agent_run_input = AgentRunInput(
                consumer_id=consumer_id,
                inputs={"agent_name": name},
                deployment=self.agent_deployments[node_index],
                signature=sign_consumer_id(consumer_id, get_private_key_from_pem(os.getenv("PRIVATE_KEY_FULL_PATH")))
            )
            
            agent = Agent()
            result = await agent.run(agent_run_input)
            logger.info(f"{name} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error running {name}: {str(e)}")
            # Return a placeholder result in case of error
            return None
        
    async def run_beauty_contest(self, module_run: OrchestratorRunInput, *args, **kwargs):
        """Run the beauty contest with agents processed in batches"""
        num_nodes = len(self.agent_deployments)
        num_agents = int(module_run.inputs.num_agents)
        agents_per_node = math.ceil(num_agents / num_nodes)

        ist = time.time()
        logger.info(f"Running {num_agents} agents in batches of {module_run.inputs.batch_size}...")
        
        all_results = []
        batch_count = math.ceil(num_agents / module_run.inputs.batch_size)
        
        for batch_num in range(batch_count):
            batch_start = batch_num * module_run.inputs.batch_size
            batch_end = min(batch_start + module_run.inputs.batch_size, num_agents)
            logger.info(f"Processing batch {batch_num+1}/{batch_count} (agents {batch_start}-{batch_end-1})")
            
            batch_tasks = []
            for i in range(batch_start, batch_end):
                node_index = min(i // agents_per_node, num_nodes - 1)
                batch_tasks.append(self.run_single_agent(i, node_index, module_run.consumer_id))
            
            # Run the batch with a timeout
            try:
                batch_timeout = 300  # 5 minutes timeout per batch
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Filter out exceptions and None results
                valid_results = [r for r in batch_results if r is not None and not isinstance(r, Exception)]
                all_results.extend(valid_results)
                
                logger.info(f"Batch {batch_num+1} completed: {len(valid_results)}/{len(batch_tasks)} agents successful")
                
                # Short pause between batches to allow system to stabilize
                if batch_num < batch_count - 1:
                    await asyncio.sleep(2)
                
            except asyncio.TimeoutError:
                logger.error(f"Batch {batch_num+1} timed out after {batch_timeout} seconds")
        
        iet = time.time()
        logger.info(f"All batches completed in {iet - ist:.2f} seconds")
        
        # Process results
        if not all_results:
            logger.error("No valid results were returned from any agents")
            return []
            
        processed_results = []
        for result in all_results:
            try:
                if result and hasattr(result, 'results') and result.results:
                    processed_results.append(result.results[0])
            except (AttributeError, IndexError) as e:
                logger.warning(f"Could not process a result: {str(e)}")
                
        logger.info(f"Final processed results count: {len(processed_results)}")
        return processed_results

async def run(module_run: Dict, *args, **kwargs):
    """Run the Keynesian beauty contest between multiple agents."""
    module_run = OrchestratorRunInput(**module_run)
    module_run.inputs = InputSchema(**module_run.inputs)
    
    beauty_contest = KeynesianBeautyContest(module_run.deployment)
    results = await beauty_contest.run_beauty_contest(module_run, *args, **kwargs)
    return results