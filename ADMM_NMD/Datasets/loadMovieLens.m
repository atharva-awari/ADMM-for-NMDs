function [R, ratings, movies, tags, links] = loadMovieLens()
% loadMovieLens  Download and load the MovieLens "latest small" dataset
%
%   [R, ratings, movies, tags, links] = loadMovieLens()
%
%   Returns:
%       R        - Sparse user–movie rating matrix (users × movies)
%       ratings  - Table of userId, movieId, rating, timestamp
%       movies   - Table of movie metadata (title, genres)
%       tags     - Table of user tags
%       links    - Table linking movieId to IMDB/Movielens IDs
%
%   The dataset (~1 MB) is downloaded automatically from GroupLens.
%
%   Example:
%       [R, ratings, movies] = loadMovieLens;
%       size(R)
%       spy(R)  % visualize sparsity pattern

    % URLs and paths
    zipUrl = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip";
    zipFile = fullfile(tempdir, "ml-latest-small.zip");
    rootDir = fullfile(tempdir, "ml-latest-small");

    % Download only if not already present
    if ~isfolder(rootDir)
        fprintf("Downloading MovieLens dataset...\n");
        websave(zipFile, zipUrl);
        unzip(zipFile, tempdir);
    else
        fprintf("Using cached dataset at %s\n", rootDir);
    end

    % Read data
    ratings = readtable(fullfile(rootDir, "ratings.csv"));
    movies  = readtable(fullfile(rootDir, "movies.csv"), 'TextType','string');
    tags    = readtable(fullfile(rootDir, "tags.csv"), 'TextType','string');
    links   = readtable(fullfile(rootDir, "links.csv"));

    % Convert timestamps to datetime (optional)
    ratings.timestamp = datetime(ratings.timestamp, 'ConvertFrom','posixtime');
    tags.timestamp = datetime(tags.timestamp, 'ConvertFrom','posixtime');

    % Map userId and movieId to continuous indices
    [~,~,ui] = unique(ratings.userId);
    [~,~,mi] = unique(ratings.movieId);

    % Build sparse user–movie matrix
    R = sparse(ui, mi, ratings.rating);

    fprintf("Users: %d | Movies: %d | Ratings: %d | Density: %.4f%%\n", ...
        size(R,1), size(R,2), nnz(R), 100*nnz(R)/numel(R));
end
