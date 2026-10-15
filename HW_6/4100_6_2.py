import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def h(s, t):
    x = (s + 1) / (s + t + 2)
    return 0.5 + 0.5 * np.sign(x - 0.5) * np.sqrt(abs(2*x - 1))

df = pd.read_csv('big12.csv')
teams = sorted(set(df.home) | set(df.away))
m = len(teams)
pos = {team: i for i, team in enumerate(teams)}

past = df[(df.season == 2025) & (df.completed == 1)]
current = df[(df.season == 2026) & (df.completed == 1) & (df.date < '2026-10-10')]
weekend = df[(df.season == 2026) & df.date.isin(['2026-10-10', '2026-10-11'])]

flat = np.full((m, m), 0.5)
last = np.full((m, m), 0.5)
total = np.zeros((m, m))
meetings = np.zeros((m, m))
wins = np.zeros(m)
played2025 = np.zeros(m)

for game in past.itertuples():
    i, j = pos[game.home], pos[game.away]
    v = h(game.home_pts, game.away_pts)
    total[i, j] += v
    total[j, i] += 1 - v
    meetings[i, j] += 1
    meetings[j, i] += 1
    played2025[i] += 1
    played2025[j] += 1
    wins[i] += game.home_pts > game.away_pts
    wins[j] += game.away_pts > game.home_pts

mask = meetings > 0
last[mask] = total[mask] / meetings[mask]
s = (wins + 1) / (played2025 + 2)
record = s[:, None] / (s[:, None] + s[None, :])

priors = {
    'Flat': flat,
    '2025 head-to-head': last,
    '2025 record': record
}
rankings = {}

for name, prior in priors.items():
    scores = np.zeros((m, m))
    games_played = np.zeros(m)

    for game in current.itertuples():
        i, j = pos[game.home], pos[game.away]
        v = h(game.home_pts, game.away_pts)
        scores[i, j] += v
        scores[j, i] += 1 - v
        games_played[i] += 1
        games_played[j] += 1

    A = (scores + prior) / (games_played[:, None] + m - 1)
    np.fill_diagonal(A, 0)

    vals, vecs = np.linalg.eig(A)
    k = np.argmax(np.abs(vals))
    r = np.real(vecs[:, k])
    if r.sum() < 0:
        r = -r
    r = r / r.sum()

    ratio = np.sort(np.abs(vals))[-2] / abs(vals[k])
    rankings[name] = r
    order = np.argsort(r)[::-1]

    print(f'\n{name}: lambda*={vals[k].real:.8f}, ratio={ratio:.8f}')
    for rank, i in enumerate(order, 1):
        print(f'{rank:2}. {teams[i]:17} {r[i]:.6f}')

    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(m), r[order])
    plt.xticks(np.arange(m), [teams[i] for i in order], rotation=65, ha='right')
    plt.xlabel('Team')
    plt.ylabel('Normalized Perron score')
    plt.title('2026 Big 12 Keener ranking: ' + name)
    plt.tight_layout()
    plt.savefig('problem2_' + name.lower().replace(' ', '_').replace('-', '_') + '.png', dpi=200)
    plt.close()

final_scores = {
    'BYU': (24, 10),
    'Oklahoma State': (42, 28),
    'West Virginia': (31, 52),
    'Kansas State': (16, 35),
    'Utah': (41, 24)
}
correct = {name: 0 for name in rankings}

print('\nOctober 10-11 predictions:')
for game in weekend.itertuples():
    home_score, away_score = final_scores[game.home]
    actual = game.home if home_score > away_score else game.away
    print(f'\n{game.away} at {game.home}: actual winner {actual}')

    for name, r in rankings.items():
        predicted = game.home if r[pos[game.home]] > r[pos[game.away]] else game.away
        correct[name] += predicted == actual
        print(f'{name}: {predicted}')

print('\nCorrect predictions:')
for name in rankings:
    print(f'{name}: {correct[name]}/{len(weekend)}')
