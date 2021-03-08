# File with metrics
from math import log2
import numpy as np

class Metrics(object):

    def precision(self, query):
        """ Calculates precision for a given query with binary relevance.

        Args:
            query (list): query with the binary relevance.

        Raises:
            ZeroDivisionError: when query is empty.

        Returns:
            float: calculated precision for the query.

        """

        # Converts input object to numpy array
        if type(query) != 'numpy.ndarray':
            query = np.asarray(query)

        # Checks that query length is greater than zero
        if query.shape[0] == 0:
            raise ZeroDivisionError("Query length is zero.")

        # Calculates precision
        else:
            return np.sum(query) / query.shape[0]

    def precision_at_k(self, query, k):
        """ Calculates precision for a given query with k relevant binary values.

        If k is greater than the number of elements in the query, the whole list
        will be used for precision calculation. If k is lower or equal to zero,
        the returned value for precision will be zero.

        Args:
            query (list): query with the binary relevant values.
            k (int): number of relevant values used for calculation.

        Raises:
            ZeroDivisionError: if the query is empty.

        Returns:
            float: calculated precision for the k relevant values.

        """

        # Converts input object to numpy array
        if type(query) != 'numpy.ndarray':
            query = np.asarray(query)

        # Checks that query is not empty
        if query.shape[0] == 0:
            raise ZeroDivisionError("Query is empty")

        # Checks that K is not lower or equal to zero.
        if k <= 0:
            return 0.0

        # Checks if K is greater than query length.
        elif k > query.shape[0]:
            return np.sum(query) / query.shape[0]

        # If K value is whithin boundaries
        else:
            return np.sum(query[0:k]) / k

    def recall_at_k(self, query, n_relevant, k):
        """ Calculates recall for a given query of a given number of relevant documents.

        If k is greater than the number of elements in the query list, the whole
        list will be used for recall calculation. If k is lower or equal to zero,
        the returned value is zero.

        Args:
            query (list): query with the binary relevant results.
            n_relevant (int): number of relevant documents used on the calculation.
            k (int): number of the top documents used on the calculation.

        Raises:
            ZeroDivisionError: if the number of relevant documents is
                zero.

        Returns
            float: recall value for the given query,

        """

        # Converts input object to numpy array
        if type(query) != 'numpy.ndarray':
            query = np.asarray(query)

        # Checks that number of relevan documents is not zero
        if n_relevant == 0:
            raise ZeroDivisionError("Number of relevant documents is zero")

        # Checks that K is not lower or equal to zero.
        if k <= 0:
            return 0

        # Checks if K is greater than query length.
        elif k > query.shape[0]:
            return np.sum(query) / n_relevant

        # If K value is whithin boundaries
        else:
            return np.sum(query[0:k]) / n_relevant

    def average_precision(self, query):
        """  Calculates average precision for a query.

        Args:
            query (list): query result with binary values.

        Raises:
            ZeroDivisionError: if query has no relevant documents.

        Returns:
            float: average-precision for the given query.

        """

        # Converts input object to numpy array
        if type(query) != 'numpy.ndarray':
            query = np.asarray(query)

        # Intializes variables to sºtore the sum of the precision and used values.
        precision_sum = 0
        precision_values = 0

        # Initializes a variable to store the previous recall value
        previous_recall = 0

        # Constant to store the number of relevant documents
        N_RELEVANT = np.sum(query)

        # Binary flag to stop iteration
        finished = False

        # Iteration variable
        k = 0

        # Iterates over the query list
        while not finished:

            # Calculates the recall at index k
            current_recall = self.recall_at_k(query, N_RELEVANT, k)

            # Checks whether the recall went up
            if current_recall > previous_recall:

                precision_sum += self.precision_at_k(query, k)
                precision_values += 1

            # Checks stop condition
            if current_recall == 1.0:
                finished = True

            # Updates variables
            previous_recall = current_recall
            k += 1

        return precision_sum / precision_values

    def MAP(self, queries):
        """ Calculates mean average precision for a set of queries.

        Args:
            quieries (list): contains vectors of queries.

        Raises:
            Exception: if list is not two-dimensional.
            ZeroDivisionError: if a query has no relevant documents.

        Returns:
            float: mean average precision for the queries.
        """


        # Converts input object to numpy array
        if type(queries) != 'numpy.ndarray':
            queries = np.asarray(queries)

        # Checks if input array is 2-dimensional
        if len(queries.shape) != 2:
            raise Exception("Input array is not two-dimensional")

        # Checks that there is at least one non-zero binary vector
        if queries.shape[1] == 0 or queries.shape[0] < 1:
            raise ZeroDivisionError("There is a zero-length binary vector")

        # Stores the number of queries used for calculating MAP
        n_queries = queries.shape[0]

        # Variable to store the sum of average-precision values
        avg_p_sum = 0

        for indx in range(n_queries):

            avg_p_sum += self.average_precision(queries[indx, :])

        return avg_p_sum / n_queries

    def DCG(self, query, k):
        """ Calculates discounted cumulative gain for a given query and index.

        Args:
            query (list): query with ranked relevance.
            k (int): index for the DCG calculation

        Returns:
            float: DCG for the given query and index.

        """

        # Converts input object to numpy array
        if type(query) != 'numpy.ndarray':
            query = np.asarray(query)

        # Returns 0 if index is lower or equal to zero or empty query
        if k <= 0 or query.shape[0] == 0:
            return 0.0

        # K is set to query size if greater than the last
        elif k > query.shape[0]:
            k = query.shape[0]

        # Intializes variable to add DCG
        dcg = 0

        # Defines a function to calculate the DCG coefficient
        def coeff(i):
            return 1.0 / log2(max([i, 2]))

        # Calculates DCG for query
        for i in range(k):
            dcg += query[i] * coeff(i + 1)

        # Returns calculated DCG
        return dcg


    def NDCG(self, query, k):
        """ Calculates normalized discounted cumulative gain.

        Args:
            query (list): query with ranked relevance
            k (int): index for DCG calculation

        Return:
            float: normalized discounted cumulative gain.

        """

        # Converts query to numpy array
        if type(query) != 'numpy.ndarray':
            query = np.asarray(query)

        # Calculates DCG for query at given index
        dcg = self.DCG(query, k)

        # Sorts query ranked relevance
        query_max = -np.sort(-query)

        # Calculates maximum DCG for query
        dcg_max = self.DCG(query_max, k)

        # Returns normalized DCG
        return dcg / dcg_max
