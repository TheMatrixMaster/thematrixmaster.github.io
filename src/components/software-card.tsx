import * as React from 'react';

import { MDXRenderer } from 'gatsby-plugin-mdx';
import { GatsbyImage, getImage, IGatsbyImageData } from 'gatsby-plugin-image';

type SoftwareCardProps = {
  id: string;
  frontmatter: {
    title: string;
    year: string;
    url: string;
    icon?: IGatsbyImageData;
  };
  body: string;
};

const SoftwareCard = ({ id, frontmatter, body }: SoftwareCardProps) => {
  const image = getImage(frontmatter.icon);
  return (
    <a key={id} className={'card'} href={frontmatter.url} target={'_blank'}>
      {image ? (
        <GatsbyImage image={image} className={'icon'} alt={frontmatter.title} />
      ) : (
        <div className={'icon'} />
      )}
      <h2>{frontmatter.title}</h2>
      <p className={'small'}>{frontmatter.year}</p>
      <MDXRenderer>{body}</MDXRenderer>
      <p className={'url small'}>{frontmatter.url}</p>
    </a>
  );
};

export default SoftwareCard;
