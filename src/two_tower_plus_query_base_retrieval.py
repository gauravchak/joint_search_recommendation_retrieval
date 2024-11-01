"""
A baseline implementation to show how to incporate query and viewer information
in embedding based retrieval.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

from src.baseline_mips_module import BaselineMIPSModule


class TwoTowerPlusQueryBaseRetrieval(nn.Module):
    """Embedding based retrieval for search and recommender systems"""

    def __init__(
        self,
        num_items: int,
        user_id_hash_size: int,
        user_id_embedding_dim: int,
        user_features_size: int,
        item_id_hash_size: int,
        item_id_embedding_dim: int,
        item_features_size: int,
        trigram_hash_size: int,
        query_size: int,
        user_value_weights: List[float],
        mips_module: BaselineMIPSModule,
        # TODO add device
    ) -> None:
        """
        Initialize the TwoTowerPlusQueryBaseRetrieval model.

        params:
            num_items: the number of items to return per user/query
            user_id_hash_size: the size of the embedding table for users
            user_id_embedding_dim (DU): internal dimension
            user_features_size (IU): input feature size for users
            item_id_hash_size: the size of the embedding table for items
            item_id_embedding_dim (DI): internal dimension
            item_features_size: (II) input feature size for items
            trigram_hash_size: size of embedding table for trigrams
            query_size: the length of substring from end of the query to use
            user_value_weights: T dimensional weights, such that a linear
                combination of point-wise immediate rewards is the best
                predictor of long term user satisfaction.
            mips_module: a module that computes the Maximum Inner Product
                Search (MIPS) over the item embeddings given the user
                embedding.
        """
        super().__init__()
        self.num_items = num_items
        # [T] dimensional vector describing how positive each label is.
        # TODO add device input.
        self.query_size: int = query_size
        self.trigram_hash_size: int = trigram_hash_size
        self.item_id_embedding_dim = item_id_embedding_dim
        self.user_value_weights = torch.tensor(user_value_weights)
        self.mips_module = mips_module

        # Create the machinery for user tower

        # 1. Create a module to represent user preference by a table lookup.
        # Please see https://github.com/gauravchak/user_preference_modeling
        # for other ways to represent user preference embedding.
        self.user_id_embedding_arch = nn.Embedding(
            user_id_hash_size, user_id_embedding_dim
        )
        # 2. Create an arch to process the user_features. We are using one
        # hidden layer of 256 dimensions. This is just a reasonable default.
        # You can experiment with other architectures.
        self.user_features_arch = nn.Sequential(
            nn.Linear(user_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, user_id_embedding_dim),
        )
        # 3. Create an arch to process the user_tower_input
        # Input dimension =
        #   user_id_embedding_dim from get_user_embedding,
        #      essentially based on user_id
        #   + user_id_embedding_dim from user_features_arch,
        #      essentially based on user_features
        #   + user_id_embedding_dim from processing the query.
        # Output dimension = item_id_embedding_dim
        # The output of this arch will be used for MIPS module.
        # Hence the output dimension needs to be same as the item tower output.
        self.user_tower_arch = nn.Linear(
            in_features=3 * user_id_embedding_dim,
            out_features=item_id_embedding_dim,
        )

        # Create the archs for item tower
        # 1. Embedding layers for item id
        self.item_id_embedding_arch = nn.Embedding(
            item_id_hash_size, item_id_embedding_dim
        )
        # 2. Create an arch to process the item_features
        self.item_features_arch = nn.Sequential(
            nn.Linear(item_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, item_id_embedding_dim),
        )
        # 3. Create an arch to process the item_tower_input
        self.item_tower_arch = nn.Linear(
            in_features=3 * item_id_embedding_dim,  # concat id and features
            out_features=item_id_embedding_dim,
        )
        # Bucketed embeddings for trigrams in query and item_name
        self.query_embedding_arch = nn.Embedding(
            trigram_hash_size, user_id_embedding_dim
        )
        self.item_name_embedding_arch = nn.Embedding(
            trigram_hash_size, item_id_embedding_dim
        )

    def get_user_embedding(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
    ) -> torch.Tensor:
        """
        Extract user representation via memorization/generalization
        The API is same as the multiple ways of user representation implemented
        in https://github.com/gauravchak/user_preference_modeling
        In particular, we recommend trying the Mixture of Represenations
        implementation in https://github.com/gauravchak/user_preference_modeling/blob/main/src/user_mo_representations.py#L62

        In this implementation we use an embedding table lookup approach.
        """
        user_id_embedding = self.user_id_embedding_arch(user_id)
        return user_id_embedding

    def process_user_features(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        user_history: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:
        """
        Process the user features to compute the input to user tower arch.

        Args:
            user_id (torch.Tensor): Tensor containing the user IDs. Shape: [B]
            user_features (torch.Tensor): Tensor containing the user features.
                Shape: [B, IU]
            user_history (torch.Tensor): For each batch an H length history of
                ids. Shape: [B, H]
                In this base implementation this is unused. In subclasses this
                affects the computation.

        Returns:
            torch.Tensor: Shape: [B, 2 * DU]
        """
        user_id_embedding = self.get_user_embedding(
            user_id=user_id, user_features=user_features
        )  # [B, DU]

        # Process user features
        user_features_embedding = self.user_features_arch(
            user_features
        )  # [B, DU]

        # Concatenate the inputs. This will be used in future to compute
        # the next user embedding.
        user_tower_input = torch.cat(
            [user_id_embedding, user_features_embedding], dim=1
        )
        return user_tower_input

    def compute_query_plus_context_embeddings(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        user_history: torch.Tensor,  # [B, H]
        query: torch.Tensor,  # [B]
    ) -> torch.Tensor:  # [B, DI]
        """
        Compute the user embedding. This will be used to query mips.

        Args:
            user_id: the user id
            user_features: the user features. We are assuming these are all
                dense features. In practice you will probably want to support
                sparse embedding features as well.
            user_history: for each user, the history of items they have
                interacted with. This is a tensor of item ids. Here we are
                assuming that the history is a fixed length, but in practice
                you will probably want to support variable length histories.
                jagged tensors are a good way to do this.
                This is NOT USED in this implementation. It is handled in a
                follow on derived class.
            query: text if present

        Returns:
            torch.Tensor: Tensor containing query + user embeddings. [B, DI]
        """
        user_tower_input = self.process_user_features(
            user_id=user_id,
            user_features=user_features,
            user_history=user_history,
        )
        query_embedding = self.compute_query_embedding(query)
        user_tower_input = torch.cat(
            [user_tower_input, query_embedding], dim=-1
        )
        # Compute the user embedding
        user_embedding = self.user_tower_arch(user_tower_input)  # [B, DI]
        return user_embedding

    def compute_item_embeddings(
        self,
        item_id: torch.Tensor,  # [B]
        item_features: torch.Tensor,  # [B, II]
        item_name: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        Process item_id, item_features, and item_name to compute embeddings.

        Args:
            item_id (torch.Tensor): Tensor containing item IDs. Shape: [B]
            item_features (torch.Tensor): Tensor containing item features.
                Shape: [B, II]
            item_name (torch.Tensor): All textual data that we want the query
                to match. Shape: [B]

        Returns:
            torch.Tensor: Tensor containing item embeddings. Shape: [B, DI]
        """
        # Process item_id
        item_id_embedding = self.item_id_embedding_arch(item_id)  # [B, D]

        # Process item_features
        item_features_embedding = self.item_features_arch(
            item_features
        )  # [B, D]

        # Process item_name
        item_name_embeddings = torch.zeros(
            (item_name.size(0), self.item_id_embedding_dim)
        )

        for i, name in enumerate(item_name):
            # Generate trigrams from item_name
            trigrams = [name[j : j + 3] for j in range(len(name) - 2)]

            # Map trigrams to ids using trigram_to_id
            trigram_ids = self.trigram_to_id(trigrams)  # [K]

            # Fetch the corresponding embeddings from item_name_embedding_arch
            trigram_embeddings = self.item_name_embedding_arch(
                trigram_ids
            )  # [K, D]

            # Mean pool the trigram embeddings to get a single embedding for
            # the item_name
            item_name_embedding = trigram_embeddings.mean(dim=0)  # [D]

            # Assign the computed embedding to item_name_embeddings tensor
            item_name_embeddings[i] = item_name_embedding

        # Concatenate the item_id, item_features, and item_name embeddings
        item_tower_input = torch.cat(
            [item_id_embedding, item_features_embedding, item_name_embeddings],
            dim=1,
        )

        # Compute the final item embedding
        item_embedding = self.item_tower_arch(item_tower_input)  # [B, DI]

        return item_embedding

    def compute_item_embeddings_batched(
        self,
        item_id: torch.Tensor,  # [B]
        item_features: torch.Tensor,  # [B, II]
        item_name: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        Process item_id, item_features, and item_name to compute embeddings in batch.
        """
        # Process item_id
        item_id_embedding = self.item_id_embedding_arch(item_id)  # [B, D]

        # Process item_features
        item_features_embedding = self.item_features_arch(
            item_features
        )  # [B, D]

        # Process item_name as batched trigrams
        all_trigrams = [
            [name[j : j + 3] for j in range(len(name) - 2)]
            for name in item_name
        ]
        flat_trigrams = [
            trigram for trigrams in all_trigrams for trigram in trigrams
        ]
        flat_trigram_ids = self.trigram_to_id(flat_trigrams)
        flat_trigram_embeddings = self.item_name_embedding_arch(
            flat_trigram_ids
        )  # [total_K, D]

        # Reconstruct item_name embeddings by mean pooling for each item
        item_name_embeddings = []
        start_idx = 0
        for trigrams in all_trigrams:
            end_idx = start_idx + len(trigrams)
            item_name_embeddings.append(
                flat_trigram_embeddings[start_idx:end_idx].mean(dim=0)
            )
            start_idx = end_idx

        # Stack into a tensor for the batch and move to device
        item_name_embeddings = torch.stack(item_name_embeddings).to(self.device)

        # Concatenate the item_id, item_features, and item_name embeddings
        item_tower_input = torch.cat(
            [item_id_embedding, item_features_embedding, item_name_embeddings],
            dim=1,
        )
        # Compute the final item embedding
        item_embedding = self.item_tower_arch(item_tower_input)  # [B, DI]
        return item_embedding

    def trigram_to_id(self, trigrams):
        # Vectorized hashing trick to map trigrams to bucket IDs
        ids = torch.tensor(
            [hash(tri) % self.trigram_hash_size for tri in trigrams]
        )
        return ids

    def compute_query_embedding(
        self,
        query: torch.Tensor,  # [B]
    ) -> torch.Tensor:  # [B, DU]
        """
        break into trigrams and fetch embeddings
        """
        query_embeddings = torch.zeros(
            query.size[0], self.user_id_embedding_dim
        )
        # Break into trigrams
        # Hash and get trigram embeddings [B, K, DU]
        # compute mean to get query embedding [B, DU]

        for i, q in enumerate(query):
            # Convert query string to lowercase and split into trigrams
            trigrams = [q[j : j + 3] for j in range(len(q) - 2)]

            # Map trigrams to ids using trigram_to_id
            trigram_ids = self.trigram_to_id(trigrams)  # [K]

            # Fetch the corresponding embeddings from query_embedding_arch
            trigram_embeddings = self.query_embedding_arch(
                trigram_ids
            )  # [K, D]

            # Mean pool the trigram embeddings to get a single embedding for
            # the query
            query_embedding = trigram_embeddings.mean(dim=0)  # [D]

            # Assign the computed embedding to the query_embeddings tensor
            query_embeddings[i] = query_embedding

        return query_embeddings  # [B, DU]

    def compute_query_embedding_batched(
        self,
        query: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        Process queries in batch to break into trigrams, fetch embeddings,
        and mean pool them to get the final query embedding.
        """
        # Step 1: Split all queries into trigrams
        # Generate trigrams for each query in batch. Use list comprehension
        # for efficiency, generating a list of lists of trigrams.
        all_trigrams = [
            [q[j : j + 3] for j in range(len(q) - 2)] for q in query
        ]  # [B, K] where K varies per query

        # Step 2: Flatten trigrams for efficient batch processing
        flat_trigrams = [
            trigram for trigrams in all_trigrams for trigram in trigrams
        ]
        flat_trigram_ids = self.trigram_to_id(
            flat_trigrams
        )  # Flattened [total_K]

        # Step 3: Look up embeddings for each trigram ID
        flat_trigram_embeddings = self.query_embedding_arch(
            flat_trigram_ids
        )  # [total_K, D]

        # Step 4: Reconstruct embedding for each query by averaging trigrams
        # Separate flat embeddings into batches based on original queries’ trigram lengths
        batch_embeddings = []
        start_idx = 0
        for trigrams in all_trigrams:
            end_idx = start_idx + len(trigrams)
            # Mean-pool embeddings of trigrams for the current query
            batch_embeddings.append(
                flat_trigram_embeddings[start_idx:end_idx].mean(dim=0)
            )  # [D]
            start_idx = end_idx

        # Convert list of tensors back to a batched tensor
        query_embeddings = torch.stack(batch_embeddings)  # [B, D]

        return query_embeddings  # [B, D]

    def forward(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        user_history: torch.Tensor,  # [B, H]
        query: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """This is used for inference.

        Compute the user embedding and return the top num_items items using the
        mips module.

        Args:
            user_id (torch.Tensor): Tensor representing the user ID. Shape: [B]
            user_features (torch.Tensor): Tensor representing the user features.
                Shape: [B, IU]
            user_history (torch.Tensor): Tensor representing the user history.
                Shape: [B, H]
            query (torch.Tensor): If present, this is the search query [B]

        Returns:
            torch.Tensor: Tensor representing the top num_items items. Shape: [B, num_items]
        """
        query_plus_context: torch.Tensor = (
            self.compute_query_plus_context_embeddings(
                user_id, user_features, user_history, query
            )
        )
        # Query the mips module to get the top num_items items and their
        # embeddings. The returned embeddings aren't strictly necessary in the
        # base implementation.
        top_items, _, _ = self.mips_module(
            query_embedding=query_plus_context, num_items=self.num_items
        )  # indices [B, num_items], mips_scores [B, NI], embeddings [B, NI, DI]  # noqa
        return top_items

    def debias_net_user_value(
        self,
        net_user_value: torch.Tensor,  # [B]
        position: torch.Tensor,  # [B]
        user_embedding: torch.Tensor,  # [B, DI]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the processed net_user_value and any losses to be added
        to the loss function.
        The idea here is to model the user value as a function of purely
        user and context features. This way the user and item interaction
        can be tasked to only predict what is incremental over what could
        have been predicted using user and position (context).

        Args:
            net_user_value (torch.Tensor): The net user value tensor [B].
            position (torch.Tensor): The position tensor of shape [B].
            user_embedding: same as what is used in MIPS  # [B, DI]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the
            processed net_user_value tensor and any losses to be added
            to the loss function.

        This is written as a function and not in train_forward to make
        it easier to implement in a derived class.
        """
        return net_user_value, 0

    def compute_training_loss(
        self,
        user_embedding: torch.Tensor,  # [B, DI]
        item_embeddings: torch.Tensor,  # [B, DI]
        position: torch.Tensor,  # [B]
        labels: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:
        # Compute the scores for every pair of user and item
        scores = torch.matmul(user_embedding, item_embeddings.t())  # [B, B]

        # You should either try to handle the popularity bias
        # of in-batch negatives using log-Q correction or
        # use random negatives.
        # [Mixed Negative Sampling paper](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/)  noqa
        # suggests random negatives is a better approach.
        # Here we are restricting ourselves to in-batch negatives and we are
        # not implementing either corrections due to time constraints.

        # Compute softmax loss
        # F.cross_entropy accepts target as
        #   ground truth class indices or class probabilities;
        # Here we are using class indices
        target = torch.arange(scores.shape[0]).to(scores.device)  # [B]

        # In the cross entropy computation below, we are not reducing
        # to mean since not every row in the batch is a "positive" example.
        # To only learn from positive examples, we are computing loss per row
        # and then using per row weights. Specifically, we are weighting the
        # loss by the net_user_value after this to give more weight to the
        # positive examples and 0 weight to the hard-negative examples.
        # Note that net_user_value is assumed to be non-negative.
        loss = F.cross_entropy(
            input=scores, target=target, reduction="none"
        )  # [B]

        # Compute the weighted average of the labels using user_value_weights
        # In the simplest case, assume you have a single label per item.
        # This label is either 1 or 0 depending on whether the user engaged
        # with this item when recommended. Then the net_user_value is 1 when
        # the user has engaged with the item and 0 otherwise.
        net_user_value = torch.matmul(labels, self.user_value_weights)  # [B]

        # Optionally debias the net_user_value by the part explained purely
        # by position. Not implemented in this version. Hence net_user_value
        # is unchanged and additional_loss is 0.
        net_user_value, additional_loss = self.debias_net_user_value(
            net_user_value=net_user_value,
            position=position,
            user_embedding=user_embedding,
        )  # [B], [1]

        # Floor by epsilon to only preserve positive net_user_value
        net_user_value = torch.clamp(
            net_user_value, min=0.000001  # small epsilon to avoid divide by 0
        )  # [B]
        # Normalize net_user_value by the max value of it in batch.
        # This is to ensure that the net_user_value is between 0 and 1.
        net_user_value = net_user_value / torch.max(net_user_value)  # [B]

        # Compute the product of loss and net_user_value
        loss = loss * net_user_value  # [B]
        loss = torch.mean(loss)  # ()

        # This loss helps us learn the debiasing archs
        loss = loss + additional_loss
        return loss

    def train_forward(
        self,
        user_id: torch.Tensor,  # [B]
        user_features: torch.Tensor,  # [B, IU]
        user_history: torch.Tensor,  # [B, H]
        query: torch.Tensor,  # [B]
        item_id: torch.Tensor,  # [B]
        item_features: torch.Tensor,  # [B, II]
        item_name: torch.Tensor,  # [B]
        position: torch.Tensor,  # [B]
        labels: torch.Tensor,  # [B, T]
    ) -> float:
        """
        This function computes the loss during training.

        Args:
            user_id (torch.Tensor): User IDs. Shape: [B].
            user_features (torch.Tensor): User features. Shape: [B, IU].
            user_history (torch.Tensor): User history. Shape: [B, H].
            query (torch.Tensor): If present in this training example, this is
                the query used to search. On recommendation training data this
                would be an empty string.
            item_id (torch.Tensor): Item IDs. Shape: [B].
            item_features (torch.Tensor): Item features. Shape: [B, II].
            item_name (torch.Tensor): This could be considered a part of item
                features but we are taking it out into its own so that it is
                easier to explain the combination with query text embedding.
            position (torch.Tensor): Position. Shape: [B].
            labels (torch.Tensor): Labels. Shape: [B, T].

        Returns:
            float: The computed loss.

        Notes:
            - The loss is computed using softmax loss and weighted by the
                net_user_value.
            - Optionally, the net_user_value can be debiased by the part
                explained purely by position.
            - The loss is clamped to preserve positive net_user_value and
                normalized between 0 and 1.
        """
        # Compute the left tower embedding
        query_plus_context: torch.Tensor = (
            self.compute_query_plus_context_embeddings(
                user_id, user_features, user_history, query
            )
        )
        # Compute item embeddings
        item_embeddings = self.compute_item_embeddings(
            item_id, item_features, item_name
        )  # [B, DI]

        loss = self.compute_training_loss(
            user_embedding=query_plus_context,
            item_embeddings=item_embeddings,
            position=position,
            labels=labels,
        )
        return loss  # ()
