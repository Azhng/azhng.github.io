---
layout: page
layout: default
title: Writings
permalink: /writings/
---


<section>

	{% for post in site.posts %}
		{% unless post.next %}
			<h3 class="code">{{ post.date | date: '%Y' }}</h3>
		{% else %}
			{% capture year %}{{ post.date | date: '%Y' }}{% endcapture %}
			{% capture nyear %}{{ post.next.date | date: '%Y' }}{% endcapture %}
			{% if year != nyear %}
				<h3 class="code">{{ post.date | date: '%Y' }}</h3>
			{% endif %}
		{% endunless %}

		{% if post.title == 'CRLDB Placeholder'%}
			<ul>
				<li>
					<div class="post-date code">
						<span>{{ post.date | date: "%b" }}</span>
					</div>
					<div class="title">
						<a href="https://www.cockroachlabs.com/blog/unordered-distinct-vectorized-engine/">Improving Unordered Distinct Efficiency in the Vectorized SQL Engine</a>
					</div>
				</li>
			</ul>
		{% else %}
			<ul>
				<li>
					<div class="post-date code">
						<span>{{ post.date | date: "%b" }}</span>
					</div>
					<div class="title">
						<a href="{{ post.url | prepend: site.baseurl | prepend: site.url }}">{{ post.title }}</a>
					</div>
				</li>
			</ul>
		{% endif %}

	{% endfor %}

</section>
