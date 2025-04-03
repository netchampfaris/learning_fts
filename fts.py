import re
import math
from collections import defaultdict
from nltk.stem import PorterStemmer


class FullTextSearch:
	def __init__(self):
		# Store for original documents
		self.documents = {}

		# Inverted index: maps words to documents containing them
		# Format: {word: [(doc_id, frequency), ...]}
		self.inverted_index = defaultdict(list)

		# Document statistics needed for scoring
		self.doc_lengths = {}  # Length of each document
		self.avg_doc_length = 0  # Average document length
		self.total_doc_count = 0  # Total number of documents

		# Initialize Porter stemmer once
		self.stemmer = PorterStemmer()

		# flags
		self.tokenize_level = None  # Tokenization level (1-4)

		# Spelling correction
		self.trigram_index = defaultdict(set)  # Maps trigrams to words containing them
		self.unique_words = set()  # Set of all unique words in the corpus

	# TOKENIZATION - Breaking text into searchable terms
	# ------------------------------------------------

	def _tokenize_basic(self, text):
		"""Step 1: Simply split text by whitespace"""
		return text.lower().split()

	def _tokenize_remove_punctuation(self, text):
		"""Step 2: Remove punctuation and split into words"""
		return re.findall(r"\w+", text.lower())

	def _tokenize_remove_stopwords(self, text):
		"""Step 3: Filter out common stop words"""
		# fmt: off
		stop_words = {"a", "an", "the", "in", "on", "at", "of", "for", "with", "by", "and", "or",
			"to", "is", "are", "was", "were", "this", "that", "be", "from", "as", "such", "which"}
		# fmt: on
		words = self._tokenize_remove_punctuation(text)
		return [word for word in words if word not in stop_words]

	def _tokenize_by_stemming(self, text):
		"""Step 4: Apply Porter stemming to normalize word forms"""
		# First tokenize and remove stop words
		words = self._tokenize_remove_stopwords(text)

		# Apply Porter stemming to each word
		return [self.stemmer.stem(word) for word in words]

	def tokenize(self, text, level=3):
		"""
		Tokenize text based on the complexity level:
		1 - Basic whitespace splitting
		2 - Remove punctuation
		3 - Remove stop words (default, most complete)
		4 - Apply Porter stemming
		"""
		if self.tokenize_level is not None:
			level = self.tokenize_level

		if level == 1:
			return self._tokenize_basic(text)
		elif level == 2:
			return self._tokenize_remove_punctuation(text)
		elif level == 3:
			return self._tokenize_remove_stopwords(text)
		else:  # level 4 or higher
			return self._tokenize_by_stemming(text)

	# INDEXING - Building the search index
	# ------------------------------------------------

	def _index_simple(self, documents):
		"""Step 1: Basic indexing without weights or normalization"""
		total_length = 0

		for doc in documents:
			doc_id = doc["id"]
			self.documents[doc_id] = doc

			# Tokenize the content
			words = self.tokenize(doc["content"], level=2)

			# Calculate document length for later
			doc_length = len(words)
			self.doc_lengths[doc_id] = doc_length
			total_length += doc_length

			# Count word frequencies
			word_freq = defaultdict(int)
			for word in words:
				word_freq[word] += 1

			# Add to inverted index
			for word, freq in word_freq.items():
				self.inverted_index[word].append((doc_id, freq))

		# Update collection statistics
		self.total_doc_count = len(documents)
		self.avg_doc_length = total_length / self.total_doc_count if self.total_doc_count > 0 else 0

		return f"Indexed {self.total_doc_count} documents with {len(self.inverted_index)} unique words"

	def _index_for_bm25(self, documents):
		"""Step 2: Indexing with title boosting"""
		total_length = 0

		for doc in documents:
			doc_id = doc["id"]
			self.documents[doc_id] = doc

			# Create a single text with title weighted more heavily
			full_text = f"{doc['title']} {doc['content']}"

			# Tokenize the combined text
			words = self.tokenize(full_text)

			# Calculate document length
			doc_length = len(words)
			self.doc_lengths[doc_id] = doc_length
			total_length += doc_length

			# Count word frequencies
			word_freq = defaultdict(int)
			for word in words:
				word_freq[word] += 1

			# Add to inverted index
			for word, freq in word_freq.items():
				self.inverted_index[word].append((doc_id, freq))

		# Update collection statistics
		self.total_doc_count = len(documents)
		self.avg_doc_length = total_length / self.total_doc_count if self.total_doc_count > 0 else 0

		return f"Indexed {self.total_doc_count} documents with {len(self.inverted_index)} unique words"

	def index_documents(self, documents, method=2):
		"""
		Build the search index from documents based on chosen method:
		1 - Simple indexing (content only)
		2 - With title boosting (default)
		"""
		# Clear previous index
		self.documents = {}
		self.inverted_index = defaultdict(list)
		self.doc_lengths = {}

		if method == 1:
			result = self._index_simple(documents)
		else:
			result = self._index_for_bm25(documents)

		# Also build spelling correction index
		self._build_spelling_index(documents)

		print(result)

	# SPELLING CORRECTION - Trigram based fuzzy matching
	# ------------------------------------------------

	def _generate_trigrams(self, word):
		"""Generate character-level trigrams for a word with padding"""
		word = f" {word} "
		return [word[i:i+3] for i in range(len(word)-2)]

	def _build_spelling_index(self, documents):
		"""Build trigram index for spelling correction from all words in documents"""
		# Clear previous index
		self.trigram_index = defaultdict(set)
		self.unique_words = set()

		# Extract all words from documents
		for doc in documents:
			# Get words from title and content
			text = f"{doc['title']} {doc['content']}"
			words = re.findall(r'\b\w+\b', text.lower())

			# Add words to trigram index
			for word in words:
				if len(word) > 2:  # Only index words with 3+ characters
					self.unique_words.add(word)
					trigrams = self._generate_trigrams(word)
					for trigram in trigrams:
						self.trigram_index[trigram].add(word)

		print(f"Built spelling correction index with {len(self.unique_words)} unique words")

	def _find_similar_words(self, word, max_suggestions=3):
		"""Find words similar to the given word using trigram similarity"""
		if len(word) < 3:
			return []

		# Generate trigrams for the input word
		query_trigrams = self._generate_trigrams(word)

		# Find candidate words that share trigrams
		candidates = set()
		for trigram in query_trigrams:
			candidates.update(self.trigram_index[trigram])

		if word in candidates:
			candidates.remove(word)  # Remove exact match

		if not candidates:
			return []

		# Calculate Jaccard similarity for each candidate
		similarities = []
		query_trigram_set = set(query_trigrams)

		for candidate in candidates:
			candidate_trigram_set = set(self._generate_trigrams(candidate))

			# Jaccard similarity: intersection / union
			intersection = len(query_trigram_set.intersection(candidate_trigram_set))
			union = len(query_trigram_set.union(candidate_trigram_set))
			similarity = intersection / union if union > 0 else 0

			# Add length similarity factor to prefer words of similar length
			len_diff = abs(len(word) - len(candidate)) / max(len(word), len(candidate))
			adjusted_similarity = similarity * (1 - 0.5 * len_diff)

			similarities.append((candidate, adjusted_similarity))

		# Sort by similarity and return top suggestions
		similarities.sort(key=lambda x: x[1], reverse=True)
		return [word for word, score in similarities[:max_suggestions]]

	def correct_query(self, query, min_word_length=4, verbose=False):
		"""Correct spelling in a query string"""
		if not self.trigram_index:
			if verbose:
				print("No spelling index available, skipping correction")
			return {"original": query,"corrected": query,"has_corrections": False}

		words = query.lower().split()
		corrected_words = []
		corrections_made = False

		if verbose:
			print(f"\nCORRECTION PROCESS FOR: '{query}'")
			print("=" * 50)

		for word in words:
			# Only try to correct words longer than min_word_length
			if len(word) >= min_word_length and word not in self.unique_words:
				if verbose:
					print(f"Word '{word}' not found in index, attempting correction...")

				suggestions = self._find_similar_words(word)

				if suggestions:
					if verbose:
						print(f"  Top suggestions: {suggestions}")
						print(f"  Correcting '{word}' → '{suggestions[0]}'")

					corrected_words.append(suggestions[0])  # Use the top suggestion
					corrections_made = True
				else:
					if verbose:
						print(f"  No suggestions found for '{word}', keeping as is")
					corrected_words.append(word)
			else:
				if verbose:
					if len(word) < min_word_length:
						print(f"Word '{word}' too short (min {min_word_length}), skipping correction")
					else:
						print(f"Word '{word}' found in index, no correction needed")
				corrected_words.append(word)

		corrected_query = " ".join(corrected_words)

		if verbose:
			print("=" * 50)
			if corrections_made:
				print(f"Final correction: '{query}' → '{corrected_query}'")
			else:
				print(f"No corrections made, query unchanged: '{query}'")
			print()

		return {
			"original": query,
			"corrected": corrected_query,
			"has_corrections": corrections_made
		}

	# SEARCH - Finding and ranking documents
	# ------------------------------------------------

	def _search_basic(self, query):
		"""Step 1: Basic search with simple term matching"""
		query_words = self.tokenize(query, level=2)
		results = []

		# Find documents containing any query word
		matching_docs = set()
		for word in query_words:
			if word in self.inverted_index:
				for doc_id, _ in self.inverted_index[word]:
					matching_docs.add(doc_id)

		# Prepare results
		for doc_id in matching_docs:
			doc = self.documents[doc_id]
			results.append(
				{
					"id": doc_id,
					"title": doc["title"],
					"snippet": doc["content"][:100] + "..." if len(doc["content"]) > 100 else doc["content"],
				}
			)

		return results

	def _calculate_tfidf_score(self, word, doc_id):
		"""
		Calculate TF-IDF score for a given word in a document.
		This provides an alternative to BM25 for comparison purposes.

		TF-IDF = Term Frequency * Inverse Document Frequency
		"""
		# How many documents contain this word
		docs_with_word = self.inverted_index[word]
		docs_count = len(docs_with_word)

		# Find the frequency of the word in this specific document
		freq = None
		for d_id, f in docs_with_word:
			if d_id == doc_id:
				freq = f
				break

		if freq is None:
			return 0  # Word not found in this document

		# Calculate Term Frequency (TF)
		# Normalize by document length to account for document size
		doc_len = self.doc_lengths[doc_id]
		tf = freq / doc_len if doc_len > 0 else 0

		# Calculate Inverse Document Frequency (IDF)
		# Add 1 to numerator and denominator to avoid division by zero
		idf = math.log((self.total_doc_count + 1) / (docs_count + 1)) + 1

		# Calculate TF-IDF score
		tfidf_score = tf * idf

		return tfidf_score

	def _calculate_term_frequency(self, query, doc_id):
		"""Calculate the combined term frequency for all query terms in a document"""
		query_words = self.tokenize(query)
		if not query_words:
			return 0

		total_tf = 0
		for word in query_words:
			if word not in self.inverted_index:
				continue

			# Find the frequency of the word in this document
			freq = 0
			for d_id, f in self.inverted_index[word]:
				if d_id == doc_id:
					freq = f
					break

			# Calculate and add term frequency
			doc_len = self.doc_lengths[doc_id]
			tf = freq / doc_len if doc_len > 0 else 0
			total_tf += tf

		return total_tf

	def _calculate_bm25_score(self, word, doc_id, k1=1.2, b=0.75):
		"""Calculate BM25 score for a given word in a document"""
		# How many documents contain this word
		docs_with_word = self.inverted_index[word]
		docs_count = len(docs_with_word)

			# Find the frequency of the word in this specific document
		freq = None
		for d_id, f in docs_with_word:
			if d_id == doc_id:
				freq = f
				break

		if freq is None:
			return 0  # Word not found in this document

		# Inverse Document Frequency (IDF) - rare words are more valuable
		idf = math.log((self.total_doc_count - docs_count + 0.5) / (docs_count + 0.5) + 1)

		# Term Frequency (TF) score with BM25 normalization
		doc_len = self.doc_lengths[doc_id]
		numerator = freq * (k1 + 1)
		denominator = freq + k1 * (1 - b + b * (doc_len / self.avg_doc_length))
		bm25_score = idf * (numerator / denominator)

		return bm25_score

	def _calculate_bm25f_score(self, word, doc_id, field_weights=None, k1=1.2, field_b=None):
		"""
		Calculate BM25F score for a word in a document with field weights.
		BM25F extends BM25 by handling multiple document fields with different weights.

		Args:
			word: The word to calculate the score for
			doc_id: The document ID
			field_weights: Dict mapping field names to weights (e.g., {"title": 2.5, "content": 1.0})
			k1: Term frequency saturation parameter
			field_b: Dict mapping field names to length normalization parameters
		"""
		# Default weights if not provided
		if field_weights is None:
			field_weights = {"title": 2.5, "content": 1.0}

		# Default b values if not provided
		if field_b is None:
			field_b = {"title": 0.75, "content": 0.75}

		# How many documents contain this word
		if word not in self.inverted_index:
			return 0

		docs_with_word = self.inverted_index[word]
		docs_count = len(docs_with_word)

		# Get the document
		if doc_id not in self.documents:
			return 0

		doc = self.documents[doc_id]

		# Calculate field-specific term frequencies
		field_tf = {}
		for field in field_weights:
			if field not in doc:
				field_tf[field] = 0
				continue

			# Count occurrences in this specific field
			field_content = doc[field]
			field_tokens = self.tokenize(field_content)
			field_tf[field] = field_tokens.count(word)

		# Calculate weighted term frequency across fields
		weighted_tf = 0
		for field, weight in field_weights.items():
			if field not in field_tf:
				continue

			# Get field length and average field length
			# (In a real implementation, we'd store these separately)
			field_len = len(self.tokenize(doc.get(field, "")))
			avg_field_len = self.avg_doc_length  # Using global average as an approximation

			# Apply field-specific normalization
			field_b_value = field_b.get(field, 0.75)
			normalized_tf = field_tf[field] / (1 - field_b_value + field_b_value * (field_len / avg_field_len))

			# Add weighted normalized term frequency
			weighted_tf += weight * normalized_tf

		# Inverse Document Frequency (IDF) calculation - same as BM25
		idf = math.log((self.total_doc_count - docs_count + 0.5) / (docs_count + 0.5) + 1)

		# Final BM25F score calculation
		numerator = weighted_tf * (k1 + 1)
		denominator = weighted_tf + k1
		bm25f_score = idf * (numerator / denominator)

		return bm25f_score

	def _get_bm25_scores(self, query):
		query_words = self.tokenize(query)

		if not query_words:
			return []

		# BM25 parameters
		k1 = 1.2  # Term frequency saturation
		b = 0.75  # Length normalization

		# Calculate scores for each document
		doc_scores = defaultdict(float)

		for word in query_words:
			if word not in self.inverted_index:
				continue

			for doc_id, _ in self.inverted_index[word]:
				# Calculate BM25 score using the extracted method
				bm25_score = self._calculate_bm25_score(word, doc_id, k1, b)

				# Add to document score
				doc_scores[doc_id] += bm25_score

		return doc_scores

	def _get_matched_words(self, query):
		query_words = self.tokenize(query)
		if not query_words:
			return []

		matched_words = defaultdict(set)

		for word in query_words:
			if word not in self.inverted_index:
				continue

			for doc_id, _ in self.inverted_index[word]:
				# Calculate BM25 score using the extracted method
				matched_words[doc_id].add(word)

		return matched_words

	def _prepare_search_results(self, doc_scores, matched_words):
		results = []
		for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
			doc = self.documents[doc_id]
			content = doc["content"]

			results.append(
				{
					"id": doc_id,
					"title": doc["title"],
					"snippet": self._create_snippet(content, matched_words[doc_id]),
					"score": round(score, 3),
				}
			)
		return results

	def _search_with_bm25(self, query):
		"""Step 2: Search with BM25 ranking algorithm"""

		print(f"Query words: {query}")

		doc_scores = self._get_bm25_scores(query)
		matched_words = self._get_matched_words(query)

		results = self._prepare_search_results(doc_scores, matched_words)
		return results

	def _create_snippet(self, content, matched_words, max_length=200):
		"""Create a snippet showing context around matched words"""
		if not matched_words:
			return content[:max_length] + ("..." if len(content) > max_length else "")

		# Find all occurrences of matched words
		matches = []

		# Check if we're using stemming (tokenize level 4)
		stemming_used = self.tokenize_level == 4

		# Find matches of words or their stemmed equivalents
		for word in matched_words:
			if stemming_used:
				# Find all words in content that would stem to our matched word
				# We need to find word boundaries and check each word
				words_in_content = re.findall(r'\b\w+\b', content.lower())
				for content_word in words_in_content:
					if self.stemmer.stem(content_word) == word:
						# Search for this original form in the content
						for match in re.finditer(r'\b' + re.escape(content_word) + r'\b', content, re.IGNORECASE):
							matches.append((match.start(), match.end(), content_word))
			else:
				# Normal exact matching
				pattern = re.compile(f"\\b{re.escape(word)}\\b", re.IGNORECASE)
				for match in pattern.finditer(content):
					matches.append((match.start(), match.end(), word))

		# Sort matches by position
		matches.sort()
		if not matches:
			return content[:max_length] + ("..." if len(content) > max_length else "")

		# Create snippets with context around matches
		context_size = 30  # Characters of context around each match
		snippets = []
		current_snippet_start = None
		current_snippet_end = None

		for start, end, word in matches:
			# Start a new snippet or extend current one
			if current_snippet_start is None or start > current_snippet_end + context_size:
				# Add current snippet if it exists
				if current_snippet_start is not None:
					snippet_text = content[max(0, current_snippet_start):min(len(content), current_snippet_end)]
					snippets.append(snippet_text)

				# Start new snippet
				current_snippet_start = max(0, start - context_size)
				current_snippet_end = min(len(content), end + context_size)
			else:
				# Extend current snippet
				current_snippet_end = min(len(content), end + context_size)

		# Add the last snippet
		if current_snippet_start is not None:
			snippet_text = content[max(0, current_snippet_start):min(len(content), current_snippet_end)]
			snippets.append(snippet_text)

		# Join snippets and limit length
		result = " ... ".join(snippets)
		if len(result) > max_length:
			result = result[:max_length] + "..."

		# Add highlighting (for original word forms if stemming was used)
		# We need to highlight all the original words we found
		words_to_highlight = {word for _, _, word in matches}
		for word in words_to_highlight:
			pattern = re.compile(f"\\b{re.escape(word)}\\b", re.IGNORECASE)
			result = pattern.sub(f"**{word}**", result)

		return result

	def search(self, query, method=3, auto_correct=True):
		"""
		Search for documents matching the query based on chosen method:
		1 - Basic search (simple term matching)
		2 - BM25 ranking without highlighting
		3 - BM25 ranking with snippets and highlighting (default)

		If auto_correct is True, will attempt to correct spelling in query
		"""
		correction_info = None
		search_query = query

		# Apply spelling correction if requested
		if auto_correct:
			correction_info = self.correct_query(query)
			if correction_info["has_corrections"]:
				search_query = correction_info["corrected"]
				print(f"Search query corrected to: '{search_query}'")

		# Perform search using corrected query if available
		if method == 1:
			results = self._search_basic(search_query)
		elif method == 2:
			results = self._search_with_bm25(search_query)
		else:  # method 3
			query_words = self.tokenize(search_query)

			if not query_words:
				return []

			# BM25 parameters
			k1 = 1.2  # Term frequency saturation
			b = 0.75  # Length normalization

			# Calculate scores for each document
			doc_scores = defaultdict(float)
			matched_words = defaultdict(set)

			for word in query_words:
				if word not in self.inverted_index:
					continue

				for doc_id, _ in self.inverted_index[word]:
					# Calculate BM25 score using the extracted method
					bm25_score = self._calculate_bm25_score(word, doc_id, k1, b)

					# Add to document score
					doc_scores[doc_id] += bm25_score
					matched_words[doc_id].add(word)

			# Prepare search results
			results = []
			for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
				doc = self.documents[doc_id]
				content = doc["content"]
				snippet = self._create_snippet(content, matched_words[doc_id])

				results.append(
					{"id": doc_id, "title": doc["title"], "snippet": snippet, "score": round(score, 3)}
				)

		# Add correction info to results if spelling was corrected
		if correction_info and correction_info["has_corrections"]:
			for result in results:
				result["correction_info"] = correction_info

		return results

	def _count_query_occurrences(self, query, field, doc):
		"""Count how many times query terms appear in the given field"""
		if field not in doc:
			return 0

		query_words = self.tokenize(query)
		field_text = doc[field]
		field_tokens = self.tokenize(field_text)

		count = 0
		for word in query_words:
			count += field_tokens.count(word)

		return count

	def _search_with_tfidf(self, query):
		"""Search with TF-IDF scoring algorithm"""
		query_words = self.tokenize(query)

		if not query_words:
			return []

		# Calculate scores for each document
		doc_scores = defaultdict(float)
		matched_words = defaultdict(set)

		for word in query_words:
			if word not in self.inverted_index:
				continue

			for doc_id, _ in self.inverted_index[word]:
				# Calculate TF-IDF score
				tfidf_score = self._calculate_tfidf_score(word, doc_id)

				# Add to document score
				doc_scores[doc_id] += tfidf_score
				matched_words[doc_id].add(word)

		# Prepare search results
		results = []
		for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
			doc = self.documents[doc_id]

			results.append({
				"id": doc_id,
				"title": doc["title"],
				"score": round(score, 3),
			})

		return results

	def _search_with_bm25f(self, query):
		"""Search with BM25F ranking algorithm"""
		query_words = self.tokenize(query)

		if not query_words:
			return []

		# BM25F parameters
		k1 = 1.2
		field_weights = {"title": 2.5, "content": 1.0}
		field_b = {"title": 0.75, "content": 0.75}

		# Calculate scores for each document
		doc_scores = defaultdict(float)
		matched_words = defaultdict(set)

		for word in query_words:
			if word not in self.inverted_index:
				continue

			for doc_id, _ in self.inverted_index[word]:
				# Calculate BM25F score
				bm25f_score = self._calculate_bm25f_score(word, doc_id, field_weights, k1, field_b)

				# Add to document score
				doc_scores[doc_id] += bm25f_score
				matched_words[doc_id].add(word)

		# Prepare search results
		results = []
		for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
			doc = self.documents[doc_id]

			results.append({
				"id": doc_id,
				"title": doc["title"],
				"score": round(score, 3),
			})

		return results

	def compare_search(self, query, methods=None, auto_correct=True):
		"""
		Compare search results using different scoring methods:
		- TF-IDF
		- BM25
		- BM25F

		Args:
			query: The search query string
			methods: List of methods to compare (e.g., ["tfidf", "bm25"]), default is all methods
			auto_correct: Whether to apply spelling correction to the query

		Results are displayed in a table format with columns for each selected method.
		"""
		search_query = query

		if methods is None:
			methods = ["tfidf", "bm25", "bm25f"]

		# Validate methods
		valid_methods = ["tfidf", "bm25", "bm25f"]
		methods = [m.lower() for m in methods if m.lower() in valid_methods]

		if not methods:
			print("No valid methods specified. Please use one or more of: tfidf, bm25, bm25f")
			return

		# Apply spelling correction if requested
		if auto_correct:
			correction_info = self.correct_query(query)
			if correction_info["has_corrections"]:
				search_query = correction_info["corrected"]
				print(f"Search query corrected to: '{search_query}'")

		# Get results from selected methods
		results = {}
		rankings = {}

		if "tfidf" in methods:
			results["tfidf"] = self._search_with_tfidf(search_query)
			rankings["tfidf"] = {result["id"]: i+1 for i, result in enumerate(results["tfidf"])}

		if "bm25" in methods:
			results["bm25"] = self._search_with_bm25(search_query)
			rankings["bm25"] = {result["id"]: i+1 for i, result in enumerate(results["bm25"])}

		if "bm25f" in methods:
			results["bm25f"] = self._search_with_bm25f(search_query)
			rankings["bm25f"] = {result["id"]: i+1 for i, result in enumerate(results["bm25f"])}

		# Create a combined result set with all document IDs
		all_doc_ids = set()
		for method in methods:
			all_doc_ids.update(result["id"] for result in results[method])

		# Create a dictionary to store all scores for each document
		combined_scores = {doc_id: {method: 0 for method in methods} for doc_id in all_doc_ids}

		# Add scores for each method
		for method in methods:
			for result in results[method]:
				doc_id = result["id"]
				combined_scores[doc_id][method] = result["score"]

		# Print the combined results in a single table
		print("\nComparison of Search Methods")
		print("===========================\n")

		# Create header row with dynamic columns based on methods
		base_columns = f"{'ID':<5} {'Title':<25} {'TF':<8} {'Title Hits':<12} {'Content Hits':<15} {'Content Len':<12}"
		method_columns = ""
		for method in methods:
			method_display = method.upper()
			method_columns += f" {method_display:<16}"

		header_width = len(base_columns) + len(method_columns)
		print("-" * header_width)
		print(base_columns + method_columns)
		print("-" * header_width)

		# Sort by the sum of selected method scores
		sorted_docs = sorted(
			combined_scores.items(),
			key=lambda x: sum(x[1].values()),
			reverse=True
		)

		# Helper function to format rank as ordinal (1st, 2nd, 3rd, etc.)
		def format_ordinal(n):
			if n == 0:
				return "-"

			if 10 <= n % 100 <= 20:
				suffix = 'th'
			else:
				suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
			return f"{n}{suffix}"

		for doc_id, scores in sorted_docs:
			doc = self.documents[doc_id]
			title_hits = self._count_query_occurrences(search_query, "title", doc)
			content_hits = self._count_query_occurrences(search_query, "content", doc)

			# Calculate and display term frequency
			tf = self._calculate_term_frequency(search_query, doc_id)
			tf_display = f"{tf:.4f}"

			# Calculate content length - token count after tokenization
			content_length = len(self.tokenize(doc["content"]))

			# Base columns
			row = f"{doc_id:<5} {doc['title'][:23]:<25} {tf_display:<8} {title_hits:<12} {content_hits:<15} {content_length:<12}"

			# Method-specific columns
			for method in methods:
				score = scores[method]
				rank = format_ordinal(rankings[method].get(doc_id, 0))
				display = f"{score:.3f} ({rank})" if score > 0 else "-"
				row += f" {display:<16}"

			print(row)

		print("-" * header_width)
		print("\nNote: Higher scores generally indicate better relevance.")
		print("TF shows the normalized term frequency (sum for all query terms).")
		print("Content Len shows token count after tokenization (words excluding stopwords).")

		# return a dictionary with results if needed for programmatic use
		return {
			"combined_scores": combined_scores,
			"results": results
		}


DOCUMENTS = [
    # DOCUMENTS FOR "full text search" - SHOWING BM25 ADVANTAGE OVER TF-IDF
    {
        "id": "1",
        "title": "Term Frequency Example with Artificial Intelligence",
        "modified": "2023-11-15 14:30:22.000000",
        "content": (
            "Full text search full text search full text search. " * 4 +
            "This document contains extreme repetition of the phrase that should be "
            "heavily overvalued by TF-IDF but properly handled by BM25's term saturation mechanism. " * 4 +
            "Artificial intelligence is mentioned here for completeness but is not the main focus."
        )
    },
    {
        "id": "2",
        "title": "Concise Search Guide",
        "modified": "2023-10-27 09:15:08.000000",
        "content": "Full text search is a powerful technique for finding relevant documents. "
                  "This very short document is highly relevant with high term density but low term count. " * 3 +
                  "Artificial intelligence tools can enhance search capabilities but are not discussed in detail here."
    },
    {
        "id": "3",
        "title": "Information Retrieval Systems",
        "modified": "2023-09-18 16:42:35.000000",
        "content": (
            "Modern information retrieval systems employ various algorithms for document ranking. "
            "Full text search capabilities are the primary component of modern retrieval systems. "
            "Full text search enables users to find relevant information quickly and efficiently. "
            "Search technologies have evolved significantly over several decades. "
            "Query processing typically involves tokenization, stemming, and term weighting. "
            "Relevance scoring considers multiple factors including term frequency and document structure. "
            "Artificial intelligence techniques are increasingly applied to improve result quality, "
            "though traditional information retrieval methods remain foundational to search systems."
        ) * 2
    },

    # DOCUMENTS FOR "artificial intelligence" - SHOWING DIFFERENCES BETWEEN BM25 AND BM25F
    {
        "id": "4",
        "title": "Artificial Intelligence Revolution in Search",
        "modified": "2023-08-05 11:23:47.000000",
        "content": "Modern computing systems are becoming increasingly sophisticated. Machine learning models "
                  "can recognize patterns in data. Artificial intelligence neural networks mimic human brain structures. "
                  "Computer vision allows machines to interpret visual information. "
                  "Artificial intelligence enhances capabilities across domains. "
                  "Full text search systems benefit from these artificial intelligence advancements, "
                  "allowing for more intuitive and accurate document retrieval."
    },
    {
        "id": "5",
        "title": "Advanced Computing Systems",  # No AI in title, many mentions in content (good for BM25F)
        "modified": "2023-07-12 08:55:19.000000",
        "content": (
            "Artificial intelligence is transforming how systems learn and make decisions. " +
            "Artificial intelligence applications span from recommendation engines to autonomous vehicles. " +
            "The rise of artificial intelligence has sparked debate about ethics and regulation. " +
            "Artificial intelligence research focuses on creating more general and capable systems. " +
            "Artificial intelligence techniques like deep learning have achieved remarkable results. " +
            "Artificial intelligence continues to evolve rapidly across many domains. " +
            "Full text search capabilities can be improved with artificial intelligence algorithms."
        )
    },
    {
        "id": "6",
        "title": "Artificial Intelligence Applications",  # AI in title (good for BM25)
        "modified": "2023-06-23 14:07:38.000000",
        "content": "Artificial intelligence is revolutionizing multiple industries. Healthcare uses AI for diagnosis "
                  "and treatment recommendations. Financial institutions employ artificial intelligence for fraud detection. "
                  "Manufacturing benefits from artificial intelligence through predictive maintenance and quality control. "
                  "Transportation is being transformed by artificial intelligence in autonomous vehicles. "
                  "Full text search systems use artificial intelligence to improve relevance scoring."
    },
    {
        "id": "7",
        "title": "Using AI for Text Search",
        "modified": "2023-05-09 10:45:51.000000",
        "content": "Artificial intelligence techniques significantly improve full text search capabilities. "
                  "Modern search engines use machine learning to rank results more effectively. "
                  "Natural language processing, a subfield of artificial intelligence, helps systems "
                  "understand search queries better. Full text search benefits from context-aware "
                  "algorithms powered by artificial intelligence. Advanced artificial intelligence "
                  "models can understand semantic relationships between search terms."
    },

    # MIXED DOCUMENTS - CONTAINING BOTH QUERY TERMS BUT LESS PROMINENTLY
    {
        "id": "8",
        "title": "Intelligent Information Retrieval",
        "modified": "2023-04-17 15:34:26.000000",
        "content": "Advanced search systems now incorporate intelligence for better results. "
                  "Some elements of full text search can be found in these systems. "
                  "Artificial intelligence concepts may be applied in limited ways. "
                  "Neural networks help categorize documents automatically. "
                  "Searching through unstructured text requires sophisticated algorithms."
    },
    {
        "id": "9",
        "title": "Search Technology Evolution",
        "modified": "2023-03-02 09:12:40.000000",
        "content": "Modern retrieval systems continue to evolve with new capabilities. "
                  "Limited artificial intelligence features are being integrated gradually. "
                  "Basic full text search functionality exists but isn't the primary focus. "
                  "This document contains both key phrases but with less prominence and frequency "
                  "compared to other documents in the collection."
    },
    {
        "id": "10",
        "title": "Programming Languages for Modern Applications",
        "modified": "2023-02-14 13:28:55.000000",
        "content": "Python is known for readability and extensive libraries. JavaScript dominates web development. "
                  "Java remains popular for enterprise applications. C++ offers performance for system programming. "
                  "SQL is essential for database operations. Each language has specific strengths for different uses. "
                  "Some are used in artificial intelligence projects. Others may implement full text search features. "
                  "Neither artificial intelligence nor full text search are central topics in this document."
    }
]