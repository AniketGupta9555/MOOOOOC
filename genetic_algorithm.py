import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class GeneticAlgorithm:
    def __init__(self, courses_df, user_preferences, num_recommendations=3, population_size=10, generations=20, mutation_rate=0.05, crossover_rate=0.7):
        self.courses_df = courses_df
        self.user_preferences = user_preferences
        self.num_recommendations = num_recommendations  # Number of recommendations (can be dynamic)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.course_indices = list(range(len(courses_df)))
        self.best_solution = None
        self.best_fitness = float('-inf')

    def fitness(self, individual):
        recommended_courses = self.courses_df.iloc[individual]
        similarities = []
        for _, course in recommended_courses.iterrows():
            course_skills = course['all_skill']
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([self.user_preferences, course_skills])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            similarities.append(similarity[0][0])
        return np.mean(similarities)

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            while True:
                individual = random.sample(self.course_indices, self.num_recommendations)  # Use num_recommendations instead of hardcoded 3
                course_names = self.courses_df.iloc[individual]['Course Name']
                if course_names.nunique() == self.num_recommendations:  # Ensure unique course names
                    break
            population.append(individual)
        return population

    def select_parents(self, population):
        tournament_size = 3
        selected_parents = []
        for _ in range(2):
            tournament = random.sample(population, tournament_size)
            tournament = sorted(tournament, key=lambda x: self.fitness(x), reverse=True)
            selected_parents.append(tournament[0])
        return selected_parents

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)
            offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
            offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
            return offspring1, offspring2
        return parent1, parent2

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            mutation_point = random.randint(0, len(individual) - 1)
            mutation_value = random.choice(self.course_indices)
            individual[mutation_point] = mutation_value
        return individual

    def run(self):
        population = self.initialize_population()
        for generation in range(self.generations):
            print(f"Generation {generation + 1}")
            population = sorted(population, key=lambda x: self.fitness(x), reverse=True)

            best_individual = population[0]
            best_individual_fitness = self.fitness(best_individual)
            if best_individual_fitness > self.best_fitness:
                self.best_solution = best_individual
                self.best_fitness = best_individual_fitness

            next_generation = population[:2]
            while len(next_generation) < self.population_size:
                parent1, parent2 = self.select_parents(population)
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)

                for offspring in [offspring1, offspring2]:
                    course_names = self.courses_df.iloc[offspring]['Course Name']
                    if course_names.nunique() == self.num_recommendations:  # Ensure unique course names
                        next_generation.append(offspring)
                    if len(next_generation) >= self.population_size:
                        break

            population = next_generation

        return self.best_solution
