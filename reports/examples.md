# Qualitative examples (MovieLens 25M, seed 42)

Sample: 5,000 random eligible users. Models: Popularity, Item CF, User CF, MF (biased), One-class. Cases are picked by the stated rules with a seeded random draw among all users meeting the rule (the count is shown), not by hand. ✓ = rated ≥ 4 in the held-out test split, ✗ = rated < 4 there, · = no held-out rating (unknown, not a dislike; the recommendation may still be good).

## How common is each case?

| Case | Users meeting the rule | Share of sample |
|:--|---:|---:|
| S1 success, heavy user with mainstream taste | 118 | 2.4% |
| S2 success, light user (<30 ratings), popularity misses | 67 | 1.3% |
| S3 success, niche-taste user | 89 | 1.8% |
| F1 failure, every model gets zero hits | 144 | 2.9% |
| F2 failure, popularity gets more hits than one-class | 127 | 2.5% |
| F2b strict version: popularity >= 3 hits, one-class none | 0 | 0.0% |
| F3 failure, MF misses where the other personalised models succeed | 117 | 2.3% |

Share of sampled users with **no hit in the top 10**: Popularity 59%, Item CF 45%, User CF 39%, MF (biased) 73%, One-class 29%.

## S1 success, heavy user with mainstream taste

**User 19217** — 580 training ratings, 31 held-out liked movies, niche score 0.08 (0 = watches only the most-rated movies, 1 = only the least-rated).

*Favourite genres in history:* Drama (130), Action (88), Thriller (83), Comedy (73).  *Genres of held-out liked movies:* Drama (16), Thriller (9), Sci-Fi (9), Comedy (8).

*Highest-rated training movies:* Wallace & Gromit: A Close Shave (1995) (5); Wallace & Gromit: The Wrong Trousers (1993) (5); Serenity (2005) (5); Million Dollar Baby (2004) (5); Shaun of the Dead (2004) (5); Shine (1996) (5); Shallow Grave (1994) (5); Lord of the Rings: The Return of the King, The (2003) (5).

*Held-out liked movies:* Good Morning, Vietnam (1987); Pi (1998); Altered States (1980); Galaxy Quest (1999); Reservoir Dogs (1992); Sixth Sense, The (1999); Gallipoli (1981); Full Metal Jacket (1987); Black Hawk Down (2001); English Patient, The (1996); Shakespeare in Love (1998); Clerks (1994) ….

| # | Popularity (3 hits) | Item CF (6 hits) | User CF (3 hits) | MF (biased) (1 hits) | One-class (5 hits) |
|--:|:--|:--|:--|:--|:--|
| 1 | · Schindler's List (1993) <sub>[48,886]</sub> | ✓ Full Metal Jacket (1987) <sub>[20,980]</sub> | · Alien (1979) <sub>[29,274]</sub> | · Memento (2000) <sub>[33,353]</sub> | ✓ Reservoir Dogs (1992) <sub>[27,987]</sub> |
| 2 | · Apollo 13 (1995) <sub>[39,175]</sub> | ✓ Reservoir Dogs (1992) <sub>[27,987]</sub> | ✓ Reservoir Dogs (1992) <sub>[27,987]</sub> | · Spirited Away (Sen to Chihiro no kamikakushi) (2001) <sub>[18,384]</sub> | ✓ Full Metal Jacket (1987) <sub>[20,980]</sub> |
| 3 | ✓ Sixth Sense, The (1999) <sub>[37,798]</sub> | · Memento (2000) <sub>[33,353]</sub> | ✓ Full Metal Jacket (1987) <sub>[20,980]</sub> | · Planet Earth II (2016) <sub>[924]</sub> | · Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) <sub>[22,639]</sub> |
| 4 | · Aladdin (1992) <sub>[35,109]</sub> | · Alien (1979) <sub>[29,274]</sub> | · Memento (2000) <sub>[33,353]</sub> | · Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001) <sub>[27,633]</sub> | ✓ Brazil (1985) <sub>[12,446]</sub> |
| 5 | · Memento (2000) <sub>[33,353]</sub> | · Schindler's List (1993) <sub>[48,886]</sub> | · Shining, The (1980) <sub>[24,102]</sub> | · Ran (1985) <sub>[4,196]</sub> | · Alien (1979) <sub>[29,274]</sub> |
| 6 | · Inception (2010) <sub>[31,468]</sub> | · Shining, The (1980) <sub>[24,102]</sub> | · Schindler's List (1993) <sub>[48,886]</sub> | · My Neighbor Totoro (Tonari no Totoro) (1988) <sub>[7,513]</sub> | ✓ Unforgiven (1992) <sub>[11,877]</sub> |
| 7 | ✓ Good Will Hunting (1997) <sub>[31,412]</sub> | ✓ Indiana Jones and the Last Crusade (1989) <sub>[30,616]</sub> | · Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001) <sub>[27,633]</sub> | · Band of Brothers (2001) <sub>[1,118]</sub> | · Memento (2000) <sub>[33,353]</sub> |
| 8 | ✓ Indiana Jones and the Last Crusade (1989) <sub>[30,616]</sub> | ✓ Sixth Sense, The (1999) <sub>[37,798]</sub> | · Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) <sub>[22,639]</sub> | ✓ Pi (1998) <sub>[8,030]</sub> | · Schindler's List (1993) <sub>[48,886]</sub> |
| 9 | · Alien (1979) <sub>[29,274]</sub> | ✓ Unforgiven (1992) <sub>[11,877]</sub> | ✓ Unforgiven (1992) <sub>[11,877]</sub> | · Princess Mononoke (Mononoke-hime) (1997) <sub>[10,565]</sub> | · Apollo 13 (1995) <sub>[39,175]</sub> |
| 10 | · Mask, The (1994) <sub>[28,002]</sub> | ✓ Brazil (1985) <sub>[12,446]</sub> | · Eternal Sunshine of the Spotless Mind (2004) <sub>[26,840]</sub> | · The Blue Planet (2001) <sub>[461]</sub> | ✓ Clerks (1994) <sub>[19,457]</sub> |

<sub>[n] = number of training ratings of the movie (popularity).</sub>

## S2 success, light user (<30 ratings), popularity misses

**User 134982** — 29 training ratings, 3 held-out liked movies, niche score 0.10 (0 = watches only the most-rated movies, 1 = only the least-rated).

*Favourite genres in history:* Drama (11), Thriller (10), Comedy (4), Romance (2).  *Genres of held-out liked movies:* Thriller (2), Drama (2), Adventure (1), Action (1).

*Highest-rated training movies:* Boys (1996) (5); Before and After (1996) (5); Sleepers (1996) (5); William Shakespeare's Romeo + Juliet (1996) (5); Eye for an Eye (1996) (5); Jerry Maguire (1996) (5); Craft, The (1996) (5); Trainspotting (1996) (5).

*Held-out liked movies:* Executive Decision (1996); Dead Man Walking (1995); Time to Kill, A (1996).

| # | Popularity (0 hits) | Item CF (0 hits) | User CF (2 hits) | MF (biased) (0 hits) | One-class (2 hits) |
|--:|:--|:--|:--|:--|:--|
| 1 | · Forrest Gump (1994) <sub>[66,003]</sub> | · Reservoir Dogs (1992) <sub>[27,987]</sub> | · Fargo (1996) <sub>[38,609]</sub> | · Planet Earth II (2016) <sub>[924]</sub> | ✓ Time to Kill, A (1996) <sub>[9,128]</sub> |
| 2 | · Shawshank Redemption, The (1994) <sub>[65,905]</sub> | · Pulp Fiction (1994) <sub>[64,375]</sub> | · Star Wars: Episode IV - A New Hope (1977) <sub>[55,599]</sub> | · Planet Earth (2006) <sub>[1,396]</sub> | · Phenomenon (1996) <sub>[10,113]</sub> |
| 3 | · Pulp Fiction (1994) <sub>[64,375]</sub> | · Clockwork Orange, A (1971) <sub>[24,145]</sub> | ✓ Dead Man Walking (1995) <sub>[15,600]</sub> | · It's Such a Beautiful Day (2011) <sub>[98]</sub> | ✓ Dead Man Walking (1995) <sub>[15,600]</sub> |
| 4 | · Silence of the Lambs, The (1991) <sub>[59,749]</sub> | · Fight Club (1999) <sub>[47,641]</sub> | · Godfather, The (1972) <sub>[42,497]</sub> | · The Blue Planet (2001) <sub>[461]</sub> | · Ransom (1996) <sub>[11,171]</sub> |
| 5 | · Matrix, The (1999) <sub>[58,845]</sub> | · Lock, Stock & Two Smoking Barrels (1998) <sub>[15,543]</sub> | · Leaving Las Vegas (1995) <sub>[16,321]</sub> | · Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001) <sub>[27,633]</sub> | · Primal Fear (1996) <sub>[8,855]</sub> |
| 6 | · Star Wars: Episode IV - A New Hope (1977) <sub>[55,599]</sub> | · Full Metal Jacket (1987) <sub>[20,980]</sub> | · Toy Story (1995) <sub>[46,422]</sub> | · Make Way for Tomorrow (1937) <sub>[97]</sub> | · Mission: Impossible (1996) <sub>[29,889]</sub> |
| 7 | · Jurassic Park (1993) <sub>[51,879]</sub> | · Snatch (2000) <sub>[20,251]</sub> | · Twelve Monkeys (a.k.a. 12 Monkeys) (1995) <sub>[38,260]</sub> | · Human Condition II, The (Ningen no joken II) (1959) <sub>[74]</sub> | · Toy Story (1995) <sub>[46,422]</sub> |
| 8 | · Schindler's List (1993) <sub>[48,886]</sub> | · Fargo (1996) <sub>[38,609]</sub> | ✓ Time to Kill, A (1996) <sub>[9,128]</sub> | · Over the Garden Wall (2013) <sub>[452]</sub> | · Leaving Las Vegas (1995) <sub>[16,321]</sub> |
| 9 | · Braveheart (1995) <sub>[47,988]</sub> | · Taxi Driver (1976) <sub>[23,430]</sub> | · English Patient, The (1996) <sub>[11,860]</sub> | · Cosmos <sub>[219]</sub> | · Twelve Monkeys (a.k.a. 12 Monkeys) (1995) <sub>[38,260]</sub> |
| 10 | · Fight Club (1999) <sub>[47,641]</sub> | · Usual Suspects, The (1995) <sub>[44,735]</sub> | · Star Wars: Episode VI - Return of the Jedi (1983) <sub>[44,475]</sub> | · Death Note: Desu nôto (2006–2007) <sub>[151]</sub> | · Eraser (1996) <sub>[12,515]</sub> |

<sub>[n] = number of training ratings of the movie (popularity).</sub>

## S3 success, niche-taste user

**User 101648** — 365 training ratings, 31 held-out liked movies, niche score 0.19 (0 = watches only the most-rated movies, 1 = only the least-rated).

*Favourite genres in history:* Comedy (87), Action (83), Adventure (71), Drama (51).  *Genres of held-out liked movies:* Drama (13), Adventure (12), Action (12), Comedy (9).

*Highest-rated training movies:* Rocky IV (1985) (5); Shrek (2001) (5); Cool Hand Luke (1967) (5); South Park: Bigger, Longer and Uncut (1999) (5); Goodfellas (1990) (5); Shawshank Redemption, The (1994) (5); Avengers, The (2012) (5); Wolf of Wall Street, The (2013) (5).

*Held-out liked movies:* War Dogs (2016); Winnie the Pooh (2011); Green Lantern: First Flight (2009); The Revenant (2015); Deadpool (2016); 42 (2013); Harry Potter and the Prisoner of Azkaban (2004); Lord of the Rings: The Return of the King, The (2003); Harry Potter and the Deathly Hallows: Part 2 (2011); Django Unchained (2012); Jurassic World (2015); Departed, The (2006) ….

| # | Popularity (1 hits) | Item CF (2 hits) | User CF (4 hits) | MF (biased) (0 hits) | One-class (4 hits) |
|--:|:--|:--|:--|:--|:--|
| 1 | · Silence of the Lambs, The (1991) <sub>[59,749]</sub> | · Godfather, The (1972) <sub>[42,497]</sub> | · Star Wars: Episode V - The Empire Strikes Back (1980) <sub>[46,415]</sub> | · Godfather, The (1972) <sub>[42,497]</sub> | ✓ Django Unchained (2012) <sub>[16,738]</sub> |
| 2 | · Star Wars: Episode IV - A New Hope (1977) <sub>[55,599]</sub> | · Silence of the Lambs, The (1991) <sub>[59,749]</sub> | ✓ Django Unchained (2012) <sub>[16,738]</sub> | · Godfather: Part II, The (1974) <sub>[27,465]</sub> | ✓ Deadpool (2016) <sub>[11,236]</sub> |
| 3 | · Jurassic Park (1993) <sub>[51,879]</sub> | · Seven (a.k.a. Se7en) (1995) <sub>[40,861]</sub> | ✓ Lord of the Rings: The Fellowship of the Ring, The (2001) <sub>[45,105]</sub> | · Band of Brothers (2001) <sub>[1,118]</sub> | · Kingsman: The Secret Service (2015) <sub>[7,694]</sub> |
| 4 | · Schindler's List (1993) <sub>[48,886]</sub> | · One Flew Over the Cuckoo's Nest (1975) <sub>[29,098]</sub> | · Godfather, The (1972) <sub>[42,497]</sub> | · The Adventures of Sherlock Holmes and Doctor Watson <sub>[120]</sub> | · Toy Story (1995) <sub>[46,422]</sub> |
| 5 | · Braveheart (1995) <sub>[47,988]</sub> | · Godfather: Part II, The (1974) <sub>[27,465]</sub> | ✓ Lord of the Rings: The Return of the King, The (2003) <sub>[41,000]</sub> | · Avengers: Infinity War - Part I (2018) <sub>[4,898]</sub> | · Sherlock Holmes (2009) <sub>[11,749]</sub> |
| 6 | · Toy Story (1995) <sub>[46,422]</sub> | ✓ Lord of the Rings: The Fellowship of the Ring, The (2001) <sub>[45,105]</sub> | · Seven (a.k.a. Se7en) (1995) <sub>[40,861]</sub> | · The Fear of 13 (2015) <sub>[36]</sub> | ✓ Batman Begins (2005) <sub>[24,681]</sub> |
| 7 | · Star Wars: Episode V - The Empire Strikes Back (1980) <sub>[46,415]</sub> | · Star Wars: Episode V - The Empire Strikes Back (1980) <sub>[46,415]</sub> | ✓ Departed, The (2006) <sub>[20,500]</sub> | · Planet Earth (2006) <sub>[1,396]</sub> | · Skyfall (2012) <sub>[8,304]</sub> |
| 8 | ✓ Lord of the Rings: The Fellowship of the Ring, The (2001) <sub>[45,105]</sub> | ✓ Batman Begins (2005) <sub>[24,681]</sub> | · Star Wars: Episode IV - A New Hope (1977) <sub>[55,599]</sub> | · You Don't Know Bo (2012) <sub>[43]</sub> | · Sherlock Holmes: A Game of Shadows (2011) <sub>[7,081]</sub> |
| 9 | · Star Wars: Episode VI - Return of the Jedi (1983) <sub>[44,475]</sub> | · Aliens (1986) <sub>[25,619]</sub> | · Gladiator (2000) <sub>[36,190]</sub> | · Cosmos <sub>[219]</sub> | ✓ Harry Potter and the Deathly Hallows: Part 2 (2011) <sub>[10,816]</sub> |
| 10 | · Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) <sub>[43,924]</sub> | · Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) <sub>[43,924]</sub> | · Memento (2000) <sub>[33,353]</sub> | · Planet Earth II (2016) <sub>[924]</sub> | · Kick-Ass (2010) <sub>[9,025]</sub> |

<sub>[n] = number of training ratings of the movie (popularity).</sub>

## F1 failure, every model gets zero hits

**User 61975** — 72 training ratings, 5 held-out liked movies, niche score 0.07 (0 = watches only the most-rated movies, 1 = only the least-rated).

*Favourite genres in history:* Drama (25), Comedy (17), Adventure (11), Crime (10).  *Genres of held-out liked movies:* Drama (3), Comedy (2), Adventure (2), Crime (2).

*Highest-rated training movies:* Silence of the Lambs, The (1991) (5); Shawshank Redemption, The (1994) (5); Raisin in the Sun, A (1961) (5); Traffic (2000) (5); Poseidon Adventure, The (1972) (5); Doors, The (1991) (5); Creepshow (1982) (5); American Graffiti (1973) (5).

*Held-out liked movies:* Grapes of Wrath, The (1940); My Left Foot (1989); Austin Powers: The Spy Who Shagged Me (1999); Who Framed Roger Rabbit? (1988); Dog Day Afternoon (1975).

| # | Popularity (0 hits) | Item CF (0 hits) | User CF (0 hits) | MF (biased) (0 hits) | One-class (0 hits) |
|--:|:--|:--|:--|:--|:--|
| 1 | · Forrest Gump (1994) <sub>[66,003]</sub> | · Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) <sub>[43,924]</sub> | · Pulp Fiction (1994) <sub>[64,375]</sub> | · Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) <sub>[43,924]</sub> | · Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) <sub>[43,924]</sub> |
| 2 | · Pulp Fiction (1994) <sub>[64,375]</sub> | · Usual Suspects, The (1995) <sub>[44,735]</sub> | · Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) <sub>[43,924]</sub> | · Planet Earth II (2016) <sub>[924]</sub> | · Being John Malkovich (1999) <sub>[24,040]</sub> |
| 3 | · Matrix, The (1999) <sub>[58,845]</sub> | · Star Wars: Episode VI - Return of the Jedi (1983) <sub>[44,475]</sub> | · Matrix, The (1999) <sub>[58,845]</sub> | · Planet Earth (2006) <sub>[1,396]</sub> | · Saving Private Ryan (1998) <sub>[37,794]</sub> |
| 4 | · Jurassic Park (1993) <sub>[51,879]</sub> | · Pulp Fiction (1994) <sub>[64,375]</sub> | · Saving Private Ryan (1998) <sub>[37,794]</sub> | · Band of Brothers (2001) <sub>[1,118]</sub> | · Usual Suspects, The (1995) <sub>[44,735]</sub> |
| 5 | · Braveheart (1995) <sub>[47,988]</sub> | · Matrix, The (1999) <sub>[58,845]</sub> | · Usual Suspects, The (1995) <sub>[44,735]</sub> | · 12 Angry Men (1957) <sub>[13,397]</sub> | · Election (1999) <sub>[10,676]</sub> |
| 6 | · Terminator 2: Judgment Day (1991) <sub>[46,409]</sub> | · Indiana Jones and the Last Crusade (1989) <sub>[30,616]</sub> | · Star Wars: Episode VI - Return of the Jedi (1983) <sub>[44,475]</sub> | · Star Wars: Episode VI - Return of the Jedi (1983) <sub>[44,475]</sub> | · Ferris Bueller's Day Off (1986) <sub>[21,868]</sub> |
| 7 | · Lord of the Rings: The Fellowship of the Ring, The (2001) <sub>[45,105]</sub> | · L.A. Confidential (1997) <sub>[22,489]</sub> | ✗ Back to the Future (1985) <sub>[40,093]</sub> | · Life Is Beautiful (La Vita è bella) (1997) <sub>[19,292]</sub> | ✗ Office Space (1999) <sub>[18,602]</sub> |
| 8 | · Usual Suspects, The (1995) <sub>[44,735]</sub> | · Stand by Me (1986) <sub>[18,288]</sub> | · Braveheart (1995) <sub>[47,988]</sub> | · Seven Samurai (Shichinin no samurai) (1954) <sub>[10,723]</sub> | · Stand by Me (1986) <sub>[18,288]</sub> |
| 9 | · Star Wars: Episode VI - Return of the Jedi (1983) <sub>[44,475]</sub> | · Saving Private Ryan (1998) <sub>[37,794]</sub> | · Forrest Gump (1994) <sub>[66,003]</sub> | · Blue Planet II (2017) <sub>[530]</sub> | · Taxi Driver (1976) <sub>[23,430]</sub> |
| 10 | · Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) <sub>[43,924]</sub> | · Aliens (1986) <sub>[25,619]</sub> | · Reservoir Dogs (1992) <sub>[27,987]</sub> | · Good, the Bad and the Ugly, The (Buono, il brutto, il cattivo, Il) (1966) <sub>[14,646]</sub> | · E.T. the Extra-Terrestrial (1982) <sub>[27,924]</sub> |

<sub>[n] = number of training ratings of the movie (popularity).</sub>

## F2 failure, popularity gets more hits than one-class

**User 66396** — 81 training ratings, 3 held-out liked movies, niche score 0.05 (0 = watches only the most-rated movies, 1 = only the least-rated).

*Favourite genres in history:* Drama (24), Comedy (12), Thriller (11), Action (10).  *Genres of held-out liked movies:* Action (2), War (1), Drama (1), Sci-Fi (1).

*Highest-rated training movies:* Once Were Warriors (1994) (5); Hudsucker Proxy, The (1994) (5); Mary Shelley's Frankenstein (Frankenstein) (1994) (5); Clerks (1994) (5); Twelve Monkeys (a.k.a. 12 Monkeys) (1995) (5); Star Trek: Generations (1994) (5); Shawshank Redemption, The (1994) (5); With Honors (1994) (4).

*Held-out liked movies:* Braveheart (1995); Terminator 2: Judgment Day (1991); Home Alone (1990).

| # | Popularity (2 hits) | Item CF (0 hits) | User CF (2 hits) | MF (biased) (0 hits) | One-class (1 hits) |
|--:|:--|:--|:--|:--|:--|
| 1 | · Matrix, The (1999) <sub>[58,845]</sub> | · Usual Suspects, The (1995) <sub>[44,735]</sub> | ✓ Braveheart (1995) <sub>[47,988]</sub> | · Band of Brothers (2001) <sub>[1,118]</sub> | · Crimson Tide (1995) <sub>[18,126]</sub> |
| 2 | ✓ Braveheart (1995) <sub>[47,988]</sub> | · Seven (a.k.a. Se7en) (1995) <sub>[40,861]</sub> | · Usual Suspects, The (1995) <sub>[44,735]</sub> | · Planet Earth II (2016) <sub>[924]</sub> | · Apollo 13 (1995) <sub>[39,175]</sub> |
| 3 | · Fight Club (1999) <sub>[47,641]</sub> | · Fight Club (1999) <sub>[47,641]</sub> | · Seven (a.k.a. Se7en) (1995) <sub>[40,861]</sub> | · Planet Earth (2006) <sub>[1,396]</sub> | ✗ Interview with the Vampire: The Vampire Chronicles (1994) <sub>[22,363]</sub> |
| 4 | · Toy Story (1995) <sub>[46,422]</sub> | · Trainspotting (1996) <sub>[23,213]</sub> | · Apollo 13 (1995) <sub>[39,175]</sub> | · Deadpool (2016) <sub>[11,236]</sub> | · Seven (a.k.a. Se7en) (1995) <sub>[40,861]</sub> |
| 5 | · Star Wars: Episode V - The Empire Strikes Back (1980) <sub>[46,415]</sub> | · Memento (2000) <sub>[33,353]</sub> | ✓ Terminator 2: Judgment Day (1991) <sub>[46,409]</sub> | · Inception (2010) <sub>[31,468]</sub> | · Firm, The (1993) <sub>[19,558]</sub> |
| 6 | ✓ Terminator 2: Judgment Day (1991) <sub>[46,409]</sub> | · Léon: The Professional (a.k.a. The Professional) (Léon) (1994) <sub>[27,086]</sub> | · Get Shorty (1995) <sub>[18,065]</sub> | · Interstellar (2014) <sub>[18,387]</sub> | ✓ Braveheart (1995) <sub>[47,988]</sub> |
| 7 | · Lord of the Rings: The Fellowship of the Ring, The (2001) <sub>[45,105]</sub> | · Reservoir Dogs (1992) <sub>[27,987]</sub> | · Dead Man Walking (1995) <sub>[15,600]</sub> | · Princess Mononoke (Mononoke-hime) (1997) <sub>[10,565]</sub> | · Usual Suspects, The (1995) <sub>[44,735]</sub> |
| 8 | · Usual Suspects, The (1995) <sub>[44,735]</sub> | · American History X (1998) <sub>[25,195]</sub> | · Fargo (1996) <sub>[38,609]</sub> | · Avengers: Infinity War - Part I (2018) <sub>[4,898]</sub> | · Quiz Show (1994) <sub>[15,459]</sub> |
| 9 | · Star Wars: Episode VI - Return of the Jedi (1983) <sub>[44,475]</sub> | · American Beauty (1999) <sub>[43,392]</sub> | · Babe (1995) <sub>[25,464]</sub> | · Black Mirror <sub>[379]</sub> | · While You Were Sleeping (1995) <sub>[17,863]</sub> |
| 10 | · Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) <sub>[43,924]</sub> | · Fargo (1996) <sub>[38,609]</sub> | · True Romance (1993) <sub>[12,337]</sub> | · The Fear of 13 (2015) <sub>[36]</sub> | · Tombstone (1993) <sub>[11,639]</sub> |

<sub>[n] = number of training ratings of the movie (popularity).</sub>

## F2b strict version: popularity >= 3 hits, one-class none

_No user in the sample meets this rule._

## F3 failure, MF misses where the other personalised models succeed

**User 139595** — 235 training ratings, 22 held-out liked movies, niche score 0.06 (0 = watches only the most-rated movies, 1 = only the least-rated).

*Favourite genres in history:* Comedy (80), Drama (58), Action (53), Adventure (49).  *Genres of held-out liked movies:* Drama (10), Comedy (10), Thriller (6), Adventure (5).

*Highest-rated training movies:* 28 Days Later (2002) (5); Spirited Away (Sen to Chihiro no kamikakushi) (2001) (5); Some Folks Call It a Sling Blade (1993) (5); Mighty Wind, A (2003) (5); White Christmas (1954) (5); Hedwig and the Angry Inch (2000) (5); Mary Shelley's Frankenstein (Frankenstein) (1994) (5); American History X (1998) (5).

*Held-out liked movies:* Gattaca (1997); Silence of the Lambs, The (1991); Great Muppet Caper, The (1981); Fifth Element, The (1997); Clear and Present Danger (1994); Speed (1994); Aviator, The (2004); Master and Commander: The Far Side of the World (2003); Iron Will (1994); Lion King, The (1994); Hunchback of Notre Dame, The (1996); Eternal Sunshine of the Spotless Mind (2004) ….

| # | Popularity (2 hits) | Item CF (4 hits) | User CF (3 hits) | MF (biased) (0 hits) | One-class (4 hits) |
|--:|:--|:--|:--|:--|:--|
| 1 | ✓ Forrest Gump (1994) <sub>[66,003]</sub> | · Star Wars: Episode V - The Empire Strikes Back (1980) <sub>[46,415]</sub> | · Star Wars: Episode V - The Empire Strikes Back (1980) <sub>[46,415]</sub> | · Spider-Man: Into the Spider-Verse (2018) <sub>[2,480]</sub> | · O Brother, Where Art Thou? (2000) <sub>[18,716]</sub> |
| 2 | ✓ Silence of the Lambs, The (1991) <sub>[59,749]</sub> | · Star Wars: Episode IV - A New Hope (1977) <sub>[55,599]</sub> | · Godfather, The (1972) <sub>[42,497]</sub> | · Guardians of the Galaxy (2014) <sub>[13,114]</sub> | ✓ Fifth Element, The (1997) <sub>[26,595]</sub> |
| 3 | · Star Wars: Episode IV - A New Hope (1977) <sub>[55,599]</sub> | · Back to the Future (1985) <sub>[40,093]</sub> | · Monty Python and the Holy Grail (1975) <sub>[30,439]</sub> | · Little Miss Sunshine (2006) <sub>[15,288]</sub> | ✓ Lion King, The (1994) <sub>[34,601]</sub> |
| 4 | · Schindler's List (1993) <sub>[48,886]</sub> | ✓ Forrest Gump (1994) <sub>[66,003]</sub> | · Schindler's List (1993) <sub>[48,886]</sub> | · My Neighbor Totoro (Tonari no Totoro) (1988) <sub>[7,513]</sub> | ✓ Office Space (1999) <sub>[18,602]</sub> |
| 5 | · Star Wars: Episode V - The Empire Strikes Back (1980) <sub>[46,415]</sub> | ✓ Snatch (2000) <sub>[20,251]</sub> | · Star Wars: Episode IV - A New Hope (1977) <sub>[55,599]</sub> | · Serenity (2005) <sub>[9,626]</sub> | ✓ Forrest Gump (1994) <sub>[66,003]</sub> |
| 6 | · Godfather, The (1972) <sub>[42,497]</sub> | ✓ Lion King, The (1994) <sub>[34,601]</sub> | ✓ Eternal Sunshine of the Spotless Mind (2004) <sub>[26,840]</sub> | · Back to the Future (1985) <sub>[40,093]</sub> | · Willy Wonka & the Chocolate Factory (1971) <sub>[22,781]</sub> |
| 7 | · Back to the Future (1985) <sub>[40,093]</sub> | · Incredibles, The (2004) <sub>[24,642]</sub> | · Back to the Future (1985) <sub>[40,093]</sub> | · Short Term 12 (2013) <sub>[1,155]</sub> | · Clueless (1995) <sub>[21,162]</sub> |
| 8 | · Fargo (1996) <sub>[38,609]</sub> | · Schindler's List (1993) <sub>[48,886]</sub> | ✓ Silence of the Lambs, The (1991) <sub>[59,749]</sub> | · Planet Earth II (2016) <sub>[924]</sub> | · Edward Scissorhands (1990) <sub>[19,192]</sub> |
| 9 | · Gladiator (2000) <sub>[36,190]</sub> | ✓ Eternal Sunshine of the Spotless Mind (2004) <sub>[26,840]</sub> | · L.A. Confidential (1997) <sub>[22,489]</sub> | · Cabin in the Woods, The (2012) <sub>[5,156]</sub> | · Star Wars: Episode IV - A New Hope (1977) <sub>[55,599]</sub> |
| 10 | · Batman (1989) <sub>[35,729]</sub> | · Léon: The Professional (a.k.a. The Professional) (Léon) (1994) <sub>[27,086]</sub> | ✓ Office Space (1999) <sub>[18,602]</sub> | · Toy Story 3 (2010) <sub>[11,698]</sub> | · Moulin Rouge (2001) <sub>[11,386]</sub> |

<sub>[n] = number of training ratings of the movie (popularity).</sub>
